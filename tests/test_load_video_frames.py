import os
from pathlib import Path
import pytest
import shutil
import subprocess
from typing import Any, Callable, Dict, Optional, Union
from unittest import mock

import numpy as np
from PIL import Image
from pydantic import BaseModel, ValidationError

from zamba.data.video import (
    load_video_frames,
    MegadetectorLiteYoloXConfig,
    VideoLoaderConfig,
)
from zamba.pytorch.dataloaders import FfmpegZambaVideoDataset

from conftest import ASSETS_DIR, TEST_VIDEOS_DIR


class Case(BaseModel):
    """A single test case for the load_video_frames function. Includes the fields:

    Attributes:
        name (str): A descriptive name of the test case
        parameters (dict): Arguments to load_video_frames
        expected_output (dict, optional): Describes the expected shape of the video that results
            from calling load_video_frames with the given parameters. Keys must be from
            ('frames', 'height', 'width', 'channels') and values can be integers or a callable that
            computes an integer from the original video metadata and output video shape.
        validation_error (bool, optional): Whether a ValidationError is expected from calling
            load_video_frames with the given parameters.
    """

    name: str
    parameters: Dict[str, Any] = {}
    expected_output: Optional[Dict[str, Union[int, Callable]]] = None
    validation_error: Optional[bool] = False


def assert_crop_bottom_pixels(original_video_metadata, video_shape, **kwargs):
    return original_video_metadata["height"] == (
        video_shape["height"] - kwargs["crop_bottom_pixels"]
    )


def assert_megadetector_le_total(original_video_metadata, video_shape, **kwargs):
    """Since we don't know how many frames the megadetector will find over the threshold,
    test that the number of frames returned by the megadetector is less than or equal
    to the total_frames argument.
    """
    return video_shape["frames"] <= kwargs["total_frames"]


def assert_megadetector_total_or_none(original_video_metadata, video_shape, **kwargs):
    """Some megadetector fill modes should resample among qualifying frames up to
    total_frames. In these cases, the number of frames returned should be either
    total_frames (if at least 1 qualifies) or 0 (if none qualify).
    If the threshold given to the megadetector is 2, then by default the
    min_over_threshold will be calculated to 0.
    """
    return video_shape["frames"] in [0, kwargs["total_frames"]]


def assert_no_frames_or_correct_shape(original_video_metadata, video_shape, **kwargs):
    return (video_shape["frames"] == 0) or (
        (video_shape["height"] == kwargs["frame_selection_height"])
        and (video_shape["width"] == kwargs["frame_selection_width"])
    )


test_cases = [
    Case(
        name="crop_bottom_pixels",
        parameters={"crop_bottom_pixels": 2},
        expected_output={"height": assert_crop_bottom_pixels},
    ),
    Case(
        name="crop_bottom_pixels_and_scene_threshold",
        parameters={
            "crop_bottom_pixels": 2,
            "scene_threshold": 0.05,
        },
        expected_output={"height": assert_crop_bottom_pixels},
    ),
    Case(
        name="crop_bottom_pixels_and_i_frames",
        parameters={
            "crop_bottom_pixels": 2,
            "i_frames": True,
        },
        expected_output={"height": assert_crop_bottom_pixels},
    ),
    Case(
        name="crop_bottom_pixels_and_video_height_width",
        parameters={
            "crop_bottom_pixels": 2,
            "frame_selection_height": 10,
            "frame_selection_width": 10,
        },
        expected_output={"height": 10, "width": 10},
    ),
    Case(
        name="crop_bottom_pixels_and_total_frames",
        parameters={
            "crop_bottom_pixels": 2,
            "total_frames": 10,
        },
        expected_output={"height": assert_crop_bottom_pixels, "frames": 10},
    ),
    Case(
        name="crop_bottom_pixels_and_fps",
        parameters={
            "crop_bottom_pixels": 2,
            "fps": 1,
        },
        expected_output={"height": assert_crop_bottom_pixels},
    ),
    Case(
        name="i_frames",
        parameters={"i_frames": True},
    ),
    Case(
        name="i_frames_and_scene_threshold",
        parameters={
            "i_frames": True,
            "scene_threshold": 0.05,
        },
        validation_error=True,
    ),
    Case(
        name="i_frames_and_video_height_width",
        parameters={
            "i_frames": True,
            "frame_selection_height": 10,
            "frame_selection_width": 10,
        },
        expected_output={"height": 10, "width": 10},
    ),
    Case(
        name="i_frames_and_total_frames",
        parameters={
            "i_frames": True,
            "total_frames": 10,
        },
        expected_output={"frames": 10},
    ),
    Case(
        name="i_frames_and_fps",
        parameters={
            "i_frames": True,
            "fps": 1,
        },
        validation_error=True,
    ),
    Case(
        name="scene_threshold",
        parameters={"scene_threshold": 0.05},
    ),
    Case(
        name="scene_threshold_and_video_height_width",
        parameters={
            "scene_threshold": 0.05,
            "frame_selection_height": 10,
            "frame_selection_width": 10,
        },
        expected_output={"height": 10, "width": 10},
    ),
    Case(
        name="scene_threshold_and_fps",
        parameters={
            "scene_threshold": 0.05,
            "fps": 10,
        },
        validation_error=True,
    ),
    Case(
        name="video_height_width",
        parameters={"frame_selection_height": 10, "frame_selection_width": 10},
        expected_output={"height": 10, "width": 10},
    ),
    Case(
        name="video_height_width_and_total_frames",
        parameters={"frame_selection_height": 10, "frame_selection_width": 10, "total_frames": 10},
        expected_output={"height": 10, "width": 10, "frames": 10},
    ),
    Case(
        name="video_height_width_and_fps",
        parameters={"frame_selection_height": 10, "frame_selection_width": 10, "fps": 10},
        expected_output={"height": 10, "width": 10},
    ),
    Case(
        name="total_frames",
        parameters={"total_frames": 10, "ensure_total_frames": True},
        expected_output={"frames": 10},
    ),
    Case(
        name="fps",
        parameters={"fps": 1},
    ),
    Case(
        name="early_bias",
        parameters={"early_bias": True},
        expected_output={"frames": 16},
    ),
    Case(
        name="early_bias_and_total_frames",
        parameters={"early_bias": True, "total_frames": 10},
        validation_error=True,
    ),
    Case(
        name="early_bias_and_scene_threshold",
        parameters={"early_bias": True, "scene_threshold": 0.2},
        validation_error=True,
    ),
    Case(
        name="early_bias_and_i_frames",
        parameters={"early_bias": True, "i_frames": True},
        validation_error=True,
    ),
    Case(
        name="early_bias_and_fps",
        parameters={"early_bias": True, "fps": 20},
        validation_error=True,
    ),
    Case(
        name="single_frame_index",
        parameters={"frame_indices": [10]},
        expected_output={"frames": 1},
    ),
    Case(
        name="multiple_frame_indices",
        parameters={"frame_indices": [0, 2, 10]},
        expected_output={"frames": 3},
    ),
    Case(
        name="frame_indices_and_total_frames",
        parameters={"frame_indices": [10], "total_frames": 10},
        validation_error=True,
    ),
    Case(
        name="frame_indices_and_scene_threshold",
        parameters={"frame_indices": [10], "scene_threshold": 0.2},
        validation_error=True,
    ),
    Case(
        name="frame_indices_and_i_frames",
        parameters={"frame_indices": [10], "i_frames": True},
        validation_error=True,
    ),
    Case(
        name="frame_indices_and_early_bias",
        parameters={"frame_indices": [10], "early_bias": True},
        validation_error=True,
    ),
    Case(
        name="evenly_sample",
        parameters={"total_frames": 10, "evenly_sample_total_frames": True},
        expected_output={"frames": 10},
    ),
    Case(
        name="evenly_sample_and_not_total_frames",
        parameters={"evenly_sample_total_frames": True, "total_frames": None},
        validation_error=True,
    ),
    Case(
        name="evenly_sample_and_scene_threshold",
        parameters={
            "total_frames": 10,
            "evenly_sample_total_frames": True,
            "scene_threshold": 0.2,
        },
        validation_error=True,
    ),
    Case(
        name="evenly_sample_and_i_frames",
        parameters={"total_frames": 10, "evenly_sample_total_frames": True, "i_frames": True},
        validation_error=True,
    ),
    Case(
        name="evenly_sample_and_fps",
        parameters={"total_frames": 10, "evenly_sample_total_frames": True, "fps": 20},
        validation_error=True,
    ),
    Case(
        name="evenly_sample_and_early_bias",
        parameters={"total_frames": 10, "evenly_sample_total_frames": True, "early_bias": True},
        validation_error=True,
    ),
    Case(
        name="megadetector_and_early_bias",
        parameters={"megadetector_lite_config": {"confidence": 0.25}, "early_bias": True},
        validation_error=True,
    ),
    Case(
        name="megadetector_and_evenly_sample",
        parameters={
            "megadetector_lite_config": {"confidence": 0.25},
            "total_frames": 10,
            "evenly_sample_total_frames": True,
        },
        validation_error=True,
    ),
    Case(
        name="megadetector_and_two_total_frames",
        parameters={
            "megadetector_lite_config": {"confidence": 0.01},
            "total_frames": 2,
            "ensure_total_frames": False,
            "fps": 2,
        },
        expected_output={"frames": assert_megadetector_total_or_none},
    ),
    Case(
        name="megadetector_and_video_height_width",
        parameters={
            "megadetector_lite_config": {"confidence": 0.01},
            "frame_selection_height": 50,
            "frame_selection_width": 50,
            "total_frames": 10,
            "crop_bottom_pixels": 2,
            "fps": 2,
            "ensure_total_frames": False,
        },
        expected_output={
            "height": assert_crop_bottom_pixels,
            "width": 50,
            "frames": assert_megadetector_le_total,
        },
    ),
]


def get_video_metadata():
    test_video_paths = sorted([path for path in TEST_VIDEOS_DIR.rglob("*") if path.is_file()])
    video_metadata = []
    for video_path in test_video_paths:
        frames, height, width, channels = load_video_frames(video_path).shape
        video_metadata.append(
            {
                "path": video_path,
                "frames": frames,
                "height": height,
                "width": width,
                "channels": channels,
            }
        )
    return video_metadata


video_metadata_values = get_video_metadata()


@pytest.fixture(
    params=video_metadata_values[:1],
    ids=[metadata["path"].stem for metadata in video_metadata_values[:1]],
)
def video_metadata(request):
    return request.param


@pytest.fixture(params=test_cases, ids=[case.name for case in test_cases])
def case(request):
    return request.param


def test_load_video_frames(case: Case, video_metadata: Dict[str, Any]):
    """Tests all pairs of test cases and test videos."""
    if case.validation_error:
        with pytest.raises(ValidationError):
            load_video_frames(video_metadata["path"], **case.parameters)

    else:
        video_shape = load_video_frames(video_metadata["path"], **case.parameters).shape
        video_shape = dict(zip(("frames", "height", "width", "channels"), video_shape))

        if case.expected_output is not None:
            for field, value in case.expected_output.items():
                if callable(value):
                    value(video_metadata, video_shape, **case.parameters)
                else:
                    assert video_shape[field] == value


def test_same_filename_new_kwargs(tmp_path, train_metadata):
    """Test that load_video_frames does not load the npz file if the params change."""
    cache = tmp_path / "test_cache"

    # prep labels for one video
    labels = (
        train_metadata[train_metadata.split == "train"]
        .set_index("filepath")
        .filter(regex="species")
        .head(1)
    )

    def _generate_dataset(config):
        """Return loaded video from FFmpegZambaVideoDataset."""
        return FfmpegZambaVideoDataset(annotations=labels, video_loader_config=config).__getitem__(
            index=0
        )[0]

    with mock.patch.dict(os.environ, {"VIDEO_CACHE_DIR": str(cache)}):
        # confirm cache is set in environment variable
        assert os.environ["VIDEO_CACHE_DIR"] == str(cache)

        first_load = _generate_dataset(config=VideoLoaderConfig(fps=1))
        new_params_same_name = _generate_dataset(config=VideoLoaderConfig(fps=2))
        assert first_load.shape != new_params_same_name.shape

        # check no params
        no_params_same_name = _generate_dataset(config=None)
        assert first_load.shape != new_params_same_name.shape != no_params_same_name.shape

        # multiple params in config
        first_load = _generate_dataset(config=VideoLoaderConfig(scene_threshold=0.2))
        new_params_same_name = _generate_dataset(
            config=VideoLoaderConfig(scene_threshold=0.2, crop_bottom_pixels=2)
        )
        assert first_load.shape != new_params_same_name.shape


def test_megadetector_lite_yolox_dog(tmp_path):
    dog = Image.open(ASSETS_DIR / "dog.jpg")
    blank = Image.new("RGB", dog.size, (64, 64, 64))
    total_frames = 10
    object_frame_indices = [0, 3, 4, 6]

    frame_directory = tmp_path / "dog"
    frame_directory.mkdir()
    for frame_index in range(total_frames):
        frame = dog if frame_index in object_frame_indices else blank
        frame.save(frame_directory / f"frame{frame_index:02}.jpg")

    subprocess.call(
        [
            "ffmpeg",
            "-r",
            "30",
            "-f",
            "image2",
            "-s",
            f"{dog.size[0]}x{dog.size[1]}",
            "-i",
            str(frame_directory / "frame%02d.jpg"),
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(tmp_path / "dog.mp4"),
            "-v",
            "quiet",
            "-hide_banner",
            "-y",
        ]
    )
    frames = load_video_frames(
        tmp_path / "dog.mp4", megadetector_lite_config=MegadetectorLiteYoloXConfig()
    )

    # Check that we detected the correct number of frames
    assert len(frames) == len(object_frame_indices)

    # Check that no blank frames were selected
    for frame in frames:
        assert not (frame == np.ones(frame.shape, dtype=np.uint8) * 64).all()


def test_resize_after_frame_selection():
    test_vid = TEST_VIDEOS_DIR / "data" / "raw" / "benjamin" / "04250002.MP4"
    resize_before_vlc = VideoLoaderConfig(
        frame_selection_height=10,
        frame_selection_width=12,
        ensure_total_frames=True,
        megadetector_lite_config={
            "confidence": 0.25,
            "fill_mode": "score_sorted",
            "n_frames": 16,
        },
    )
    a = load_video_frames(filepath=test_vid, config=resize_before_vlc)

    # use full size image for MDLite
    resize_after_vlc = VideoLoaderConfig(
        model_input_height=10,
        model_input_width=12,
        ensure_total_frames=True,
        megadetector_lite_config={
            "confidence": 0.25,
            "fill_mode": "score_sorted",
            "n_frames": 16,
        },
    )
    b = load_video_frames(filepath=test_vid, config=resize_after_vlc)

    # shapes should be the same
    assert a.shape == b.shape

    # but we expect some frame differences
    assert (a != b).any()


def test_validate_total_frames():
    config = VideoLoaderConfig(
        megadetector_lite_config=MegadetectorLiteYoloXConfig(confidence=0.01, n_frames=None),
        total_frames=10,
    )
    assert config.megadetector_lite_config.n_frames == 10

    config = VideoLoaderConfig(
        megadetector_lite_config=MegadetectorLiteYoloXConfig(confidence=0.01, n_frames=8),
    )
    assert config.total_frames == 8


def test_caching(tmp_path, caplog, train_metadata):
    cache = tmp_path / "video_cache"

    # prep labels for one video
    labels = (
        train_metadata[train_metadata.split == "train"]
        .set_index("filepath")
        .filter(regex="species")
        .head(1)
    )

    # no caching by default
    _ = FfmpegZambaVideoDataset(
        annotations=labels,
    ).__getitem__(index=0)
    assert not cache.exists()

    # caching can be specifed in config
    _ = FfmpegZambaVideoDataset(
        annotations=labels, video_loader_config=VideoLoaderConfig(fps=1, cache_dir=cache)
    ).__getitem__(index=0)

    # one file in cache
    assert len([f for f in cache.rglob("*") if f.is_file()]) == 1
    shutil.rmtree(cache)

    # or caching can be specified in environment variable
    with mock.patch.dict(os.environ, {"VIDEO_CACHE_DIR": str(cache)}):
        _ = FfmpegZambaVideoDataset(
            annotations=labels,
        ).__getitem__(index=0)
        assert len([f for f in cache.rglob("*") if f.is_file()]) == 1

        # changing cleanup in config does not prompt new hashing of videos
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            _ = FfmpegZambaVideoDataset(
                annotations=labels, video_loader_config=VideoLoaderConfig(cleanup_cache=True)
            ).__getitem__(index=0)

            assert "Loading from cache" in caplog.text

    # if no config is passed, this is equivalent to specifying None/False in all non-cache related VLC params
    no_config = FfmpegZambaVideoDataset(annotations=labels, video_loader_config=None).__getitem__(
        index=0
    )[0]

    config_with_nones = FfmpegZambaVideoDataset(
        annotations=labels,
        video_loader_config=VideoLoaderConfig(
            crop_bottom_pixels=None,
            i_frames=False,
            scene_threshold=None,
            megadetector_lite_config=None,
            frame_selection_height=None,
            frame_selection_width=None,
            total_frames=None,
            ensure_total_frames=False,
            fps=None,
            early_bias=False,
            frame_indices=None,
            evenly_sample_total_frames=False,
            pix_fmt="rgb24",
            model_input_height=None,
            model_input_width=None,
        ),
    ).__getitem__(index=0)[0]

    assert np.array_equal(no_config, config_with_nones)


def test_validate_video_cache_dir():
    with mock.patch.dict(os.environ, {"VIDEO_CACHE_DIR": "example_cache_dir"}):
        config = VideoLoaderConfig()
        assert config.cache_dir == Path("example_cache_dir")

    for cache in ["", 0]:
        with mock.patch.dict(os.environ, {"VIDEO_CACHE_DIR": str(cache)}):
            config = VideoLoaderConfig()
            assert config.cache_dir is None
