from enum import Enum
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pydantic import BaseModel
import torch
from tqdm import tqdm
from yolox.utils.boxes import postprocess

from zamba.object_detection import YoloXModel

LOCAL_MD_LITE_MODEL = Path(__file__).parent / "assets" / "yolox_tiny_640_20220528.pth"
LOCAL_MD_LITE_MODEL_KWARGS = (
    Path(__file__).parent / "assets" / "yolox_tiny_640_20220528_model_kwargs.json"
)


class FillModeEnum(str, Enum):
    """Enum for frame filtering fill modes

    Attributes:
        repeat: Randomly resample qualifying frames to get to n_frames
        score_sorted: Take up to n_frames in sort order (even if some have zero probability)
        weighted_euclidean: Sample the remaining frames weighted by their euclidean distance in
            time to the frames over the threshold
        weighted_prob: Sample the remaining frames weighted by their predicted probability
    """

    repeat = "repeat"
    score_sorted = "score_sorted"
    weighted_euclidean = "weighted_euclidean"
    weighted_prob = "weighted_prob"


class MegadetectorLiteYoloXConfig(BaseModel):
    """Configuration for a MegadetectorLiteYoloX frame selection model

    Attributes:
        confidence (float): Only consider object detections with this confidence or greater
        nms_threshold (float): Non-maximum suppression is a method for filtering many bounding
            boxes around the same object to a single bounding box. This is a constant that
            determines how much to suppress similar bounding boxes.
        image_width (int): Scale image to this width before sending to object detection model.
        image_height (int): Scale image to this height before sending to object detection model.
        device (str): Where to run the object detection model, "cpu" or "cuda".
        frame_batch_size (int): Number of frames to predict on at once.
        n_frames (int, optional): Max number of frames to return. If None returns all frames above
            the threshold. Defaults to None.
        fill_mode (str, optional): Mode for upsampling if the number of frames above the threshold
            is less than n_frames. Defaults to "repeat".
        sort_by_time (bool, optional): Whether to sort the selected frames by time (original order)
            before returning. If False, returns frames sorted by score (descending). Defaults to
            True.
        seed (int, optional): Random state for random number generator. Defaults to 55.
    """

    confidence: float = 0.25
    nms_threshold: float = 0.45
    image_width: int = 640
    image_height: int = 640
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    frame_batch_size: int = 24
    n_frames: Optional[int] = None
    fill_mode: Optional[FillModeEnum] = FillModeEnum.score_sorted
    sort_by_time: bool = True
    seed: Optional[int] = 55

    class Config:
        extra = "forbid"


class MegadetectorLiteYoloX:
    def __init__(
        self,
        path: os.PathLike = LOCAL_MD_LITE_MODEL,
        kwargs: os.PathLike = LOCAL_MD_LITE_MODEL_KWARGS,
        config: Optional[Union[MegadetectorLiteYoloXConfig, dict]] = None,
    ):
        """MegadetectorLite based on YOLOX.

        Args:
            path (pathlike): Path to trained YoloX model checkpoint (.pth extension)
            config (MegadetectorLiteYoloXConfig): YoloX configuration
        """
        if config is None:
            config = MegadetectorLiteYoloXConfig()
        elif isinstance(config, dict):
            config = MegadetectorLiteYoloXConfig.parse_obj(config)

        yolox = YoloXModel.load(
            checkpoint=path,
            model_kwargs_path=kwargs,
        )

        ckpt = torch.load(yolox.args.ckpt, map_location=config.device)
        model = yolox.exp.get_model()
        model.load_state_dict(ckpt["model"])
        model = model.eval().to(config.device)

        self.model = model
        self.yolox = yolox
        self.config = config
        self.num_classes = yolox.exp.num_classes

    @staticmethod
    def scale_and_pad_array(
        image_array: np.ndarray, output_width: int, output_height: int
    ) -> np.ndarray:
        return np.array(
            ImageOps.pad(
                Image.fromarray(image_array),
                (output_width, output_height),
                method=Image.BICUBIC,
                color=None,
                centering=(0, 0),
            )
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Process an image for the model, including scaling/padding the image, transposing from
        (height, width, channel) to (channel, height, width) and casting to float.
        """
        arr = np.ascontiguousarray(
            self.scale_and_pad_array(frame, self.config.image_width, self.config.image_height),
            dtype=np.float32,
        )
        return np.moveaxis(arr, 2, 0)

    def _preprocess_video(self, video: np.ndarray) -> np.ndarray:
        """Process a video for the model, including resizing the frames in the video, transposing
        from (batch, height, width, channel) to (batch, channel, height, width) and casting to float.
        """
        resized_video = np.zeros(
            (video.shape[0], video.shape[3], self.config.image_height, self.config.image_width),
            dtype=np.float32,
        )
        for frame_idx in range(video.shape[0]):
            resized_video[frame_idx] = self._preprocess(video[frame_idx])
        return resized_video

    def detect_video(self, video_arr: np.ndarray, pbar: bool = False):
        """Runs object detection on an video.

        Args:
            video_arr (np.ndarray): An video array with dimensions (frames, height, width, channels).
            pbar (int): Whether to show progress bar. Defaults to False.

        Returns:
            list: A list containing detections and score for each frame. Each tuple contains two arrays:
                the first is an array of bounding box detections with dimensions (object, 4) where
                object is the number of objects detected and the other 4 dimension are
                (x1, y1, x2, y1). The second is an array of object detection confidence scores of
                length (object) where object is the number of objects detected.
        """

        pbar = tqdm if pbar else lambda x: x

        # batch of frames
        batch_size = self.config.frame_batch_size

        video_outputs = []
        with torch.no_grad():
            for i in range(0, len(video_arr), batch_size):
                a = video_arr[i : i + batch_size]

                outputs = self.model(
                    torch.from_numpy(self._preprocess_video(a)).to(self.config.device)
                )
                outputs = postprocess(
                    outputs, self.num_classes, self.config.confidence, self.config.nms_threshold
                )
                video_outputs.extend(outputs)

        detections = []
        for o in pbar(video_outputs):
            detections.append(
                self._process_frame_output(
                    o, original_height=video_arr.shape[1], original_width=video_arr.shape[2]
                )
            )

        return detections

    def detect_image(self, img_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs object detection on an image.

        Args:
            img_arr (np.ndarray): An image array with dimensions (height, width, channels).

        Returns:
            np.ndarray: An array of bounding box detections with dimensions (object, 4) where
                object is the number of objects detected and the other 4 dimension are
                (x1, y1, x2, y1).

            np.ndarray: An array of object detection confidence scores of length (object) where
                object is the number of objects detected.
        """
        with torch.no_grad():
            outputs = self.model(
                torch.from_numpy(self._preprocess(img_arr)).unsqueeze(0).to(self.config.device)
            )
            output = postprocess(
                outputs, self.num_classes, self.config.confidence, self.config.nms_threshold
            )

        return self._process_frame_output(output[0], img_arr.shape[0], img_arr.shape[1])

    def _process_frame_output(self, output, original_height, original_width):
        if output is None:
            return np.array([]), np.array([])
        else:
            detections = pd.DataFrame(
                output.cpu().numpy(),
                columns=["x1", "y1", "x2", "y2", "score1", "score2", "class_num"],
            ).assign(score=lambda row: row.score1 * row.score2)

            # Transform bounding box to be in terms of the original image dimensions
            ratio = min(
                self.config.image_width / original_width,
                self.config.image_height / original_height,
            )
            detections[["x1", "y1", "x2", "y2"]] /= ratio

            # Express bounding boxes in terms of proportions of original image dimensions
            detections[["x1", "x2"]] /= original_width
            detections[["y1", "y2"]] /= original_height

            return detections[["x1", "y1", "x2", "y2"]].values, detections.score.values

    def filter_frames(
        self, frames: np.ndarray, detections: List[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        """Filter video frames using megadetector lite.

        Which frames are returned depends on the fill_mode and how many frames are above the
        confidence threshold. If more than n_frames are above the threshold, the top n_frames are
        returned. Otherwise add to those over threshold based on fill_mode. If none of these
        conditions are met, returns all frames above the threshold.

        Args:
            frames (np.ndarray): Array of video frames to filter with dimensions (frames, height,
                width, channels)
            detections (list of tuples): List of detection results for each frame. Each element is
                a tuple of the list of bounding boxes [array(x1, y1, x2, y2)] and the detection
                 probabilities, both as float

        Returns:
            np.ndarray: An array of video frames of length n_frames or shorter
        """

        frame_scores = pd.Series(
            [(np.max(score) if (len(score) > 0) else 0) for _, score in detections]
        ).sort_values(
            ascending=False
        )  # reduce to one score per frame

        selected_indices = frame_scores.loc[frame_scores > self.config.confidence].index

        if self.config.n_frames is None:
            # no minimum n_frames provided, just select all the frames with scores > threshold
            pass

        elif len(selected_indices) >= self.config.n_frames:
            # num. frames with scores > threshold is greater than the requested number of frames
            selected_indices = (
                frame_scores[selected_indices]
                .sort_values(ascending=False)
                .iloc[: self.config.n_frames]
                .index
            )

        elif len(selected_indices) < self.config.n_frames:
            # num. frames with scores > threshold is less than the requested number of frames
            # repeat frames that are above threshold to get to n_frames
            rng = np.random.RandomState(self.config.seed)

            if self.config.fill_mode == "repeat":
                repeated_indices = rng.choice(
                    selected_indices,
                    self.config.n_frames - len(selected_indices),
                    replace=True,
                )
                selected_indices = np.concatenate((selected_indices, repeated_indices))

            # take frames in sorted order up to n_frames, even if score is zero
            elif self.config.fill_mode == "score_sorted":
                selected_indices = (
                    frame_scores.sort_values(ascending=False).iloc[: self.config.n_frames].index
                )

            # sample up to n_frames, prefer points closer to frames with detection
            elif self.config.fill_mode == "weighted_euclidean":
                sample_from = frame_scores.loc[~frame_scores.index.isin(selected_indices)].index
                # take one over euclidean distance to all points with detection
                weights = [1 / np.linalg.norm(selected_indices - sample) for sample in sample_from]
                # normalize weights
                weights /= np.sum(weights)
                sampled = rng.choice(
                    sample_from,
                    self.config.n_frames - len(selected_indices),
                    replace=False,
                    p=weights,
                )

                selected_indices = np.concatenate((selected_indices, sampled))

            # sample up to n_frames, weight by predicted probability - only if some frames have nonzero prob
            elif (self.config.fill_mode == "weighted_prob") and (len(selected_indices) > 0):
                sample_from = frame_scores.loc[~frame_scores.index.isin(selected_indices)].index
                weights = frame_scores[sample_from] / np.sum(frame_scores[sample_from])
                sampled = rng.choice(
                    sample_from,
                    self.config.n_frames - len(selected_indices),
                    replace=False,
                    p=weights,
                )

                selected_indices = np.concatenate((selected_indices, sampled))

        # sort the selected images back into their original order
        if self.config.sort_by_time:
            selected_indices = sorted(selected_indices)

        return frames[selected_indices]
