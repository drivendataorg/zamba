from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from zamba.data.video import VideoMetadata


class Detection(BaseModel):
    pixels: List[int]  # (y1, x1, y2, x2) in pixels
    proportions: List[float]  # (y1, x1, y2, x2) in proportion of total image width or height
    box_width: int  # width in pixels
    box_height: int  # height in pixels
    score: float  # probability score

    @staticmethod
    def _proportions_to_pixels(
        proportions: Tuple[float, float, float, float], height: int, width: int
    ) -> Tuple[int, int, int, int]:
        """Converts bounding boxes from proportions to pixels.

        Args:
            proportions (tuple of float): Bounding box coordinates (y1, x1, y2, x2)
            height (int): Image height in pixels
            width (int): Image width in pixels

        Returns:
            tuple of int: Bounding box in terms of pixels (y1, x1, y2, x2)
        """
        return (
            round(height * proportions[0]),
            round(width * proportions[1]),
            round(height * proportions[2]),
            round(width * proportions[3]),
        )

    @staticmethod
    def _pixels_to_proportions(
        pixels: np.ndarray, height: int, width: int
    ) -> Tuple[float, float, float, float]:
        """Converts bounding boxes from pixels to proportions.

        Args:
            pixels (np.ndarray): Bounding box coordinates (y1, x1, y2, x2)
            height (int): Image height in pixels
            width (int): Image width in pixels

        Returns:
            tuple of int: Bounding box in terms of image proportions (y1, x1, y2, x2)
        """
        return (
            pixels[0] / height,
            pixels[1] / width,
            pixels[2] / height,
            pixels[3] / width,
        )

    @classmethod
    def from_box_score(
        cls,
        box: np.ndarray,
        score: float,
        width: int,
        height: int,
        box_is_proportion: bool = True,
    ) -> "Detection":
        """Instantiates from detection bounding boxes and scores and the image dimensions.

        Args:
            box (np.ndarray): Bounding box coordinates as (y1, x1, y2, x2)
            score (float): Object detection confidence score for the bounding box
            width (int): Image width in pixels
            height (int): Image height in pixels
            box_is_proportion (bool): If true, box coordinates are in image proportions. If false,
                box coordinates are in pixels.

        Returns:
            Detection
        """
        if box_is_proportion:
            proportions = box
            pixels = cls._proportions_to_pixels(proportions, height, width)
        else:
            pixels = box
            proportions = cls._pixels_to_proportions(pixels, height, width)

        return cls(
            proportions=list(proportions),
            pixels=list(pixels),
            box_width=pixels[3] - pixels[1],
            box_height=pixels[2] - pixels[0],
            score=score,
        )

    @staticmethod
    def _darknet_to_yx(darknet_box: Tuple[float, float, float, float]) -> "Detection":
        """(cx, cy, w, h) -> (y1, x1, y2, x2)"""
        cy, cx, h, w = darknet_box
        return [
            cy - (h / 2),
            cx - (w / 2),
            cy + (h / 2),
            cx + (w / 2),
        ]

    @classmethod
    def from_darknet(cls, detection, raw_img_size, darknet_img_size):
        name, score, (center_x, center_y, w, h) = detection

        # y1, x1, y2, x2, as proportions
        proportions = [
            round(center_y - (h / 2)) / darknet_img_size[1],
            round(center_x - (w / 2)) / darknet_img_size[0],
            round(center_y + (h / 2)) / darknet_img_size[1],
            round(center_x + (w / 2)) / darknet_img_size[0],
        ]

        pixels = cls._proportions_to_pixels(proportions, raw_img_size[0], raw_img_size[1])

        return cls(
            proportions=proportions,
            pixels=pixels,
            box_width=pixels[3] - pixels[1],
            box_height=pixels[2] - pixels[0],
            score=score,
        )


class ImageDetections(BaseModel):
    detections: List[Detection]
    frame_ix: Optional[int]
    timestamp_s: Optional[float]
    img_width: int
    img_height: int

    @classmethod
    def from_image_and_boxes(
        cls,
        image_arr: np.ndarray,
        boxes: List[np.ndarray],
        scores: List[np.ndarray],
        frame_ix: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> "ImageDetections":
        """Instantiates from an image array and detection bounding boxes and scores.

        Args:
            image_arr (np.ndarray): An image array with dimension (height, width, channels)
            boxes (list of ndarray): List of bounding boxes where each box is (y1, x1, y2, x2)

        Returns:
            ImageDetections
        """
        height, width = image_arr.shape[:2]

        detections = [
            Detection.from_box_score(box, score, width=width, height=height)
            for box, score in zip(boxes, scores)
        ]

        return cls(
            detections=detections,
            frame_ix=frame_ix,
            timestamp_s=timestamp_s,
            img_width=width,
            img_height=height,
        )

    @classmethod
    def from_darknet(
        cls,
        detections,
        raw_img_size,
        darknet_img_size,
        cls_threshold=None,
        frame_ix=None,
        timestamp_s=None,
    ):
        detection_models = [
            Detection.from_darknet(d, raw_img_size, darknet_img_size)
            for d in detections
            if cls_threshold is None
            or (d[5] >= cls_threshold)  # d=[cx, cy, bw, bh, box_prob, cls_prob]
        ]

        return cls(
            detections=detection_models,
            frame_ix=frame_ix,
            timestamp_s=timestamp_s,
            img_width=raw_img_size[0],
            img_height=raw_img_size[1],
        )

    def plot_on_image(self, frame: np.ndarray, color: str = "m"):
        fig, ax = plt.subplots()

        ax.matshow(frame)

        for detection in self.detections:
            rectangle = mpl.patches.Rectangle(
                (
                    detection.proportions[1] * self.img_width,
                    detection.proportions[0] * self.img_height,
                ),
                (detection.proportions[3] - detection.proportions[1]) * self.img_width,
                (detection.proportions[2] - detection.proportions[0]) * self.img_height,
                fill=False,
                edgecolor=color,
            )

            ax.add_patch(rectangle)


class VideoDetections(BaseModel):
    frames: List[ImageDetections]
    threshold_set: Optional[float] = None
    meta: Optional[VideoMetadata]
