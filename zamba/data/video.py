from fractions import Fraction
from functools import reduce
import hashlib
import json
from math import floor
import os
from pathlib import Path
import subprocess
from shutil import rmtree
import tempfile
from typing import Optional, Union, List

import cv2
from cloudpathlib import S3Path, AnyPath
import ffmpeg
from loguru import logger
import numpy as np
import pandas as pd
from pydantic import BaseModel, root_validator, validator

from zamba.exceptions import ZambaFfmpegException
from zamba.object_detection.yolox.megadetector_lite_yolox import (
    MegadetectorLiteYoloX,
    MegadetectorLiteYoloXConfig,
)
from zamba.settings import IMAGE_SUFFIXES


def ffprobe(path: os.PathLike) -> pd.Series:
    def flatten_json(j, name=""):
        for k in j:
            if isinstance(j[k], dict):
                yield from flatten_json(j[k], f"{name}.{k}")
            elif isinstance(j[k], list):
                for i in range(len(j[k])):
                    yield from flatten_json(j[k][i], f"{name}.{k}[{i}]")
            else:
                yield {f"{name}.{k}".strip("."): j[k]}

    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "stream:format",
            "-select_streams",
            "v",
            "-of",
            "json",
            path,
        ]
    )
    output = json.loads(output)
    result = reduce(lambda a, b: {**a, **b}, flatten_json(output))
    return pd.Series(result)


def get_video_stream(path: Union[os.PathLike, S3Path]) -> dict:
    try:
        probe = ffmpeg.probe(str(path))
    except ffmpeg.Error as exc:
        raise ZambaFfmpegException(exc.stderr)

    return next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)


def num_frames(stream_or_path: Union[dict, os.PathLike, S3Path]) -> Optional[int]:
    if not isinstance(stream_or_path, dict):
        stream = get_video_stream(stream_or_path)
    else:
        stream = stream_or_path

    if not stream:
        return

    if "nb_frames" in stream:
        return int(stream["nb_frames"])

    if "duration" in stream:
        duration = float(stream["duration"])

        if "r_frame_rate" in stream:
            frame_rate = float(Fraction(stream["r_frame_rate"]))
        elif "avg_frame_rate" in stream:
            frame_rate = float(stream["avg_frame_rate"])
        duration -= float(stream.get("start_time", 0))

        return floor(duration * frame_rate)


def ensure_frame_number(arr, total_frames: int):
    """Ensures the array contains the requested number of frames either by clipping frames from
    the end or dulpicating the last frame.

    Args:
        arr (np.ndarray): Array of video frames with shape (frames, height, width, channel).
        total_frames (int): Desired number of frames in output array.
    """
    if (total_frames is None) or (arr.shape[0] == total_frames):
        return arr
    elif arr.shape[0] == 0:
        logger.warning(
            "No frames selected. Returning an array in the desired shape with all zeros."
        )
        return np.zeros((total_frames, arr.shape[1], arr.shape[2], arr.shape[3]), dtype="int")
    elif arr.shape[0] > total_frames:
        logger.info(
            f"Clipping {arr.shape[0] - total_frames} frames "
            f"(original: {arr.shape[0]}, requested: {total_frames})."
        )
        return arr[:total_frames]
    elif arr.shape[0] < total_frames:
        logger.info(
            f"Duplicating last frame {total_frames - arr.shape[0]} times "
            f"(original: {arr.shape[0]}, requested: {total_frames})."
        )
        return np.concatenate(
            [arr, np.tile(arr[-1], (total_frames - arr.shape[0], 1, 1, 1))], axis=0
        )


def get_frame_time_estimates(path: os.PathLike):
    probe = ffmpeg.probe(str(path), show_entries="frame=best_effort_timestamp_time")
    return [float(x["best_effort_timestamp_time"]) for x in probe["frames"]]


class VideoMetadata(BaseModel):
    height: int
    width: int
    n_frames: int
    duration_s: float
    fps: int

    @classmethod
    def from_video(cls, path: os.PathLike):
        stream = get_video_stream(path)
        return cls(
            height=int(stream["height"]),
            width=int(stream["width"]),
            n_frames=int(stream["nb_frames"]),
            duration_s=float(stream["duration"]),
            fps=int(Fraction(stream["r_frame_rate"])),  # reported, not average
        )


class VideoLoaderConfig(BaseModel):
    """
    Configuration for load_video_frames.

    Args:
        crop_bottom_pixels (int, optional): Number of pixels to crop from the bottom of the video
            (prior to resizing to `video_height`).
        i_frames (bool, optional): Only load the I-Frames. See
            https://en.wikipedia.org/wiki/Video_compression_picture_types#Intra-coded_(I)_frames/slices_(key_frames)
        scene_threshold (float, optional): Only load frames that correspond to scene changes.
            See http://www.ffmpeg.org/ffmpeg-filters.html#select_002c-aselect
        megadetector_lite_config (MegadetectorLiteYoloXConfig, optional): Configuration of
            MegadetectorLiteYoloX frame selection model.
        frame_selection_height (int, optional): Resize the video to this height in pixels, prior to
            frame selection. If None, the full size video will be used for frame selection. Using full
            size images (setting to None) is recommended for MegadetectorLite, especially if your
            species of interest are smaller.
        frame_selection_width (int, optional): Resize the video to this width in pixels, prior to
            frame selection.
        total_frames (int, optional): Number of frames that should ultimately be returned.
        ensure_total_frames (bool): Selecting the number of frames by resampling may result in one
            more or fewer frames due to rounding. If True, ensure the requested number of frames
            is returned by either clipping or duplicating the final frame. Raises an error if no
            frames have been selected. Otherwise, return the array unchanged.
        fps (float, optional): Resample the video evenly from the entire duration to a specific
            number of frames per second.
        early_bias (bool, optional): Resamples to 24 fps and selects 16 frames biased toward the
            front (strategy used by competition winner).
        frame_indices (list(int), optional): Select specific frame numbers. Note: frame selection
            is done after any resampling.
        evenly_sample_total_frames (bool, optional): Reach the total number of frames specified by
            evenly sampling from the duration of the video. Defaults to False.
        pix_fmt (str, optional): ffmpeg pixel format, defaults to 'rgb24' for RGB channels; can be
            changed to 'bgr24' for BGR.
        model_input_height (int, optional): After frame selection, resize the video to this height
            in pixels.
        model_input_width (int, optional): After frame selection, resize the video to this width in
            pixels.
        cache_dir (Path, optional): Cache directory where preprocessed videos will be saved
            upon first load. Alternatively, can be set with VIDEO_CACHE_DIR environment variable.
            Defaults to None, which means videos will not be cached. Provided there is enough space
            on your machine, it is highly encouraged to cache videos for training as this will
            speed up all subsequent epochs. If you are predicting on the same videos with the
            same video loader configuration, this will save time on future runs.
        cleanup_cache (bool): Whether to delete the cache dir after training or predicting ends.
            Defaults to False.
    """

    crop_bottom_pixels: Optional[int] = None
    i_frames: Optional[bool] = False
    scene_threshold: Optional[float] = None
    megadetector_lite_config: Optional[MegadetectorLiteYoloXConfig] = None
    frame_selection_height: Optional[int] = None
    frame_selection_width: Optional[int] = None
    total_frames: Optional[int] = None
    ensure_total_frames: Optional[bool] = True
    fps: Optional[float] = None
    early_bias: Optional[bool] = False
    frame_indices: Optional[List[int]] = None
    evenly_sample_total_frames: Optional[bool] = False
    pix_fmt: Optional[str] = "rgb24"
    model_input_height: Optional[int] = None
    model_input_width: Optional[int] = None
    cache_dir: Optional[Path] = None
    cleanup_cache: bool = False

    class Config:
        extra = "forbid"

    @validator("cache_dir", always=True)
    def validate_video_cache_dir(cls, cache_dir):
        """Set up cache directory for preprocessed videos. Config argument takes precedence
        over environment variable.
        """
        if cache_dir is None:
            cache_dir = os.getenv("VIDEO_CACHE_DIR", None)

            if cache_dir in ["", "0"]:
                cache_dir = None

        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir

    @root_validator(skip_on_failure=True)
    def check_height_and_width(cls, values):
        if (values["frame_selection_height"] is None) ^ (values["frame_selection_width"] is None):
            raise ValueError(
                f"Must provide both frame_selection_height and frame_selection_width or neither. Values provided are {values}."
            )
        if (values["model_input_height"] is None) ^ (values["model_input_width"] is None):
            raise ValueError(
                f"Must provide both model_input_height and model_input_width or neither. Values provided are {values}."
            )
        return values

    @root_validator(skip_on_failure=True)
    def check_fps_compatibility(cls, values):
        if values["fps"] and (
            values["evenly_sample_total_frames"] or values["i_frames"] or values["scene_threshold"]
        ):
            raise ValueError(
                f"fps cannot be used with evenly_sample_total_frames, i_frames, or scene_threshold. Values provided are {values}."
            )
        return values

    @root_validator(skip_on_failure=True)
    def check_i_frame_compatibility(cls, values):
        if values["scene_threshold"] and values["i_frames"]:
            raise ValueError(
                f"i_frames cannot be used with scene_threshold. Values provided are {values}."
            )
        return values

    @root_validator(skip_on_failure=True)
    def check_early_bias_compatibility(cls, values):
        if values["early_bias"] and (
            values["i_frames"]
            or values["scene_threshold"]
            or values["total_frames"]
            or values["evenly_sample_total_frames"]
            or values["fps"]
        ):
            raise ValueError(
                f"early_bias cannot be used with i_frames, scene_threshold, total_frames, evenly_sample_total_frames, or fps. Values provided are {values}."
            )
        return values

    @root_validator(skip_on_failure=True)
    def check_frame_indices_compatibility(cls, values):
        if values["frame_indices"] and (
            values["total_frames"]
            or values["scene_threshold"]
            or values["i_frames"]
            or values["early_bias"]
            or values["evenly_sample_total_frames"]
        ):
            raise ValueError(
                f"frame_indices cannot be used with total_frames, scene_threshold, i_frames, early_bias, or evenly_sample_total_frames. Values provided are {values}."
            )
        return values

    @root_validator(skip_on_failure=True)
    def check_megadetector_lite_compatibility(cls, values):
        if values["megadetector_lite_config"] and (
            values["early_bias"] or values["evenly_sample_total_frames"]
        ):
            raise ValueError(
                f"megadetector_lite_config cannot be used with early_bias or evenly_sample_total_frames. Values provided are {values}."
            )
        return values

    @root_validator(skip_on_failure=True)
    def check_evenly_sample_total_frames_compatibility(cls, values):
        if values["evenly_sample_total_frames"] is True and values["total_frames"] is None:
            raise ValueError(
                f"total_frames must be specified if evenly_sample_total_frames is used. Values provided are {values}."
            )
        if values["evenly_sample_total_frames"] and (
            values["scene_threshold"]
            or values["i_frames"]
            or values["fps"]
            or values["early_bias"]
        ):
            raise ValueError(
                f"evenly_sample_total_frames cannot be used with scene_threshold, i_frames, fps, or early_bias. Values provided are {values}."
            )
        return values

    @root_validator(skip_on_failure=True)
    def validate_total_frames(cls, values):
        if values["megadetector_lite_config"] is not None:
            # set n frames for megadetector_lite_config if only specified by total_frames
            if values["megadetector_lite_config"].n_frames is None:
                values["megadetector_lite_config"].n_frames = values["total_frames"]

            # set total frames if only specified in megadetector_lite_config
            if values["total_frames"] is None:
                values["total_frames"] = values["megadetector_lite_config"].n_frames

        return values


def get_cached_array_path(vid_path, config):
    """Get the path to where the cached array would be, if it exists.

    vid_path: string path to the video, or Path
    config: VideoLoaderConfig

    returns: Path object to the cached data
    """
    assert isinstance(config, VideoLoaderConfig)

    # don't include `cleanup_cache` or `cache_dir` in the hashed config
    # NOTE: sorting the keys avoids a cache miss if we see the same config in a different order;
    # might not be necessary with a VideoLoaderConfig
    config_dict = config.dict()
    keys = config_dict.keys() - {"cleanup_cache", "cache_dir"}
    hashed_part = {k: config_dict[k] for k in sorted(keys)}

    # hash config for inclusion in path
    hash_str = hashlib.sha1(str(hashed_part).encode("utf-8")).hexdigest()
    logger.opt(lazy=True).debug(f"Generated hash {hash_str} from {hashed_part}")

    # strip leading "/" in absolute path
    vid_path = AnyPath(str(vid_path).lstrip("/"))

    # if the video is in S3, drop the prefix and bucket name
    if isinstance(vid_path, S3Path):
        vid_path = AnyPath(vid_path.key)

    cache_dir = config.cache_dir
    npy_path = AnyPath(cache_dir) / hash_str / vid_path.with_suffix(".npy")
    return npy_path


class npy_cache:
    def __init__(self, cache_path: Optional[Path] = None, cleanup: bool = False):
        self.cache_path = cache_path
        self.cleanup = cleanup

    def __call__(self, f):
        def _wrapped(*args, **kwargs):
            try:
                vid_path = kwargs["filepath"]
            except Exception:
                vid_path = args[0]
            try:
                config = kwargs["config"]
            except Exception:
                config = VideoLoaderConfig(**kwargs)

            # NOTE: what should we do if this assert fails?
            assert config.cache_dir == self.cache_path

            # get the path for the cached data
            npy_path = get_cached_array_path(vid_path, config)

            # make parent directories since we're using absolute paths
            npy_path.parent.mkdir(parents=True, exist_ok=True)

            if npy_path.exists():
                logger.debug(f"Loading from cache {npy_path}: size {npy_path.stat().st_size}")
                return np.load(npy_path)
            else:
                logger.debug(f"Loading video from disk: {vid_path}")
                loaded_video = f(*args, **kwargs)
                np.save(npy_path, loaded_video)
                logger.debug(f"Wrote to cache {npy_path}: size {npy_path.stat().st_size}")
                return loaded_video

        if self.cache_path is not None:
            return _wrapped
        else:
            return f

    def __del__(self):
        if hasattr(self, "cache_path") and self.cleanup and self.cache_path.exists():
            if self.cache_path.parents[0] == tempfile.gettempdir():
                logger.info(f"Deleting cache dir {self.cache_path}.")
                rmtree(self.cache_path)
            else:
                logger.warning(
                    "Bravely refusing to delete directory that is not a subdirectory of the "
                    "system temp directory. If you really want to delete, do so manually using:\n "
                    f"rm -r {self.cache_path}"
                )


def load_and_repeat_image(path, target_size=(224, 224), repeat_count=4):
    """
    Loads an image, resizes it, and repeats it N times.

    Args:
        path: Path to the image file.
        target_size: A tuple (w, h) representing the desired width and height of the resized image.
        repeat_count: Number of times to repeat the image.

    Returns:
        A NumPy array of shape (N, h, w, 3) representing the repeated image.
    """
    image = cv2.imread(str(path))

    # Resize the image in same way as video frames are in `load_video_frames`
    image = cv2.resize(
        image,
        target_size,
        # https://stackoverflow.com/a/51042104/1692709
        interpolation=(
            cv2.INTER_LINEAR
            if image.shape[1] < target_size[0]  # compare image width with target width
            else cv2.INTER_AREA
        ),
    )

    image_array = np.expand_dims(image, axis=0)

    # Repeat the image N times
    repeated_image = np.repeat(image_array, repeat_count, axis=0)

    return repeated_image


def load_video_frames(
    filepath: os.PathLike,
    config: Optional[VideoLoaderConfig] = None,
    **kwargs,
):
    """Loads frames from videos using fast ffmpeg commands.

    Supports images as well, but it is inefficient since we just replicate the frames.

    Args:
        filepath (os.PathLike): Path to the video.
        config (VideoLoaderConfig, optional): Configuration for video loading.
        **kwargs: Optionally, arguments for VideoLoaderConfig can be passed in directly.

    Returns:
        np.ndarray: An array of video frames with dimensions (time x height x width x channels).
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    if config is None:
        config = VideoLoaderConfig(**kwargs)

    if Path(filepath).suffix.lower() in IMAGE_SUFFIXES:
        return load_and_repeat_image(
            filepath,
            target_size=(config.model_input_width, config.model_input_height),
            repeat_count=config.total_frames,
        )

    video_stream = get_video_stream(filepath)
    w = int(video_stream["width"])
    h = int(video_stream["height"])

    pipeline = ffmpeg.input(str(filepath))
    pipeline_kwargs = {}

    if (config.crop_bottom_pixels is not None) and (config.crop_bottom_pixels > 0):
        # scale to ensure all frames are the same height and we can crop
        pipeline = pipeline.filter("scale", f"{w},{h}")
        pipeline = pipeline.crop("0", "0", "iw", f"ih-{config.crop_bottom_pixels}")
        h = h - config.crop_bottom_pixels

    if config.evenly_sample_total_frames:
        config.fps = config.total_frames / float(video_stream["duration"])

    if config.early_bias:
        config.fps = 24  # competition frame selection assumes 24 frames per second
        config.total_frames = 16  # used for ensure_total_frames

    if config.fps:
        pipeline = pipeline.filter("fps", fps=config.fps, round="up")

    if config.i_frames:
        pipeline = pipeline.filter("select", "eq(pict_type,PICT_TYPE_I)")

    if config.scene_threshold:
        pipeline = pipeline.filter("select", f"gt(scene,{config.scene_threshold})")

    if config.frame_selection_height and config.frame_selection_width:
        pipeline = pipeline.filter(
            "scale", f"{config.frame_selection_width},{config.frame_selection_height}"
        )
        w, h = config.frame_selection_width, config.frame_selection_height

    if config.early_bias:
        config.frame_indices = [2, 8, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156]

    if config.frame_indices:
        pipeline = pipeline.filter("select", "+".join(f"eq(n,{f})" for f in config.frame_indices))
        pipeline_kwargs = {"vsync": 0}

    pipeline = pipeline.output(
        "pipe:", format="rawvideo", pix_fmt=config.pix_fmt, **pipeline_kwargs
    )

    try:
        out, err = pipeline.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        raise ZambaFfmpegException(exc.stderr)

    arr = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])

    if config.megadetector_lite_config is not None:
        mdlite = MegadetectorLiteYoloX(config=config.megadetector_lite_config)
        detection_probs = mdlite.detect_video(video_arr=arr)

        arr = mdlite.filter_frames(arr, detection_probs)

    if (config.model_input_height is not None) and (config.model_input_width is not None):
        resized_frames = np.zeros(
            (arr.shape[0], config.model_input_height, config.model_input_width, 3), np.uint8
        )
        for ix, f in enumerate(arr):
            if (f.shape[0] != config.model_input_height) or (
                f.shape[1] != config.model_input_width
            ):
                f = cv2.resize(
                    f,
                    (config.model_input_width, config.model_input_height),
                    # https://stackoverflow.com/a/51042104/1692709
                    interpolation=(
                        cv2.INTER_LINEAR
                        if f.shape[1] < config.model_input_width
                        else cv2.INTER_AREA
                    ),
                )
            resized_frames[ix, ...] = f
        arr = np.array(resized_frames)

    if config.ensure_total_frames:
        arr = ensure_frame_number(arr, total_frames=config.total_frames)

    return arr
