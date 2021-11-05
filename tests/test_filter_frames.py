import numpy as np
import pytest

from zamba.object_detection.yolox.megadetector_lite_yolox import MegadetectorLiteYoloXConfig

n_frames = 100
rng = np.random.RandomState(68891)


@pytest.fixture
def frames():
    # 20 6x8 RGB frames where the upper-right hand pixel is the frame index
    frames = rng.randint(0, 255, size=(n_frames, 6, 8, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(n_frames, dtype=np.uint8)
    return frames


@pytest.fixture
def detections():
    # Frame scores that increase monotonically
    scores = np.zeros(n_frames, dtype=float)
    scores[np.arange(0, n_frames, 10)] = np.arange(0, n_frames, 10)

    # Include a second score for each frame, always smaller 1 so that when we consolidate by
    # frame only the first score matters
    scores = np.c_[scores, rng.random(n_frames)]

    boxes = rng.random(size=(n_frames, 4)) * 100

    return [(box, score) for box, score in zip(boxes, scores)]


def test_reduce_selection(mdlite, frames, detections):
    """Tests the case where object detection selects more than enough frames."""
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=10, n_frames=5, fill_mode="repeat", sort_by_time=False
    )
    filtered_frames = mdlite.filter_frames(frames, detections)
    assert (filtered_frames[:, 0, 0, 0] == np.array([90, 80, 70, 60, 50])).all()


def test_repeat(mdlite, frames, detections):
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=50, n_frames=5, fill_mode="repeat", sort_by_time=False
    )
    filtered_frames = mdlite.filter_frames(frames, detections)
    assert (filtered_frames[:, 0, 0, 0] == np.array([90, 80, 70, 60, 80])).all()


def test_score_sorted(mdlite, frames, detections):
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=50, n_frames=5, fill_mode="score_sorted", sort_by_time=False
    )
    filtered_frames = mdlite.filter_frames(frames, detections)
    assert (filtered_frames[:, 0, 0, 0] == np.array([90, 80, 70, 60, 50])).all()


def test_sort_by_time(mdlite, frames, detections):
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=50, n_frames=5, fill_mode="repeat", sort_by_time=True
    )
    filtered_frames = mdlite.filter_frames(frames, detections)
    assert (filtered_frames[:, 0, 0, 0] == np.array([60, 70, 80, 80, 90])).all()


def test_weighted_euclidean(mdlite, frames, detections):
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=50, n_frames=20, fill_mode="weighted_euclidean", sort_by_time=False
    )
    filtered_frames = mdlite.filter_frames(frames, detections)
    assert (
        filtered_frames[:, 0, 0, 0]
        == np.array(
            [90, 80, 70, 60, 44, 67, 73, 5, 54, 64, 65, 34, 93, 72, 56, 50, 87, 83, 47, 88]
        )
    ).all()


def test_weighted_prob(mdlite, frames, detections):
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=50, n_frames=20, fill_mode="weighted_prob", sort_by_time=False
    )
    filtered_frames = mdlite.filter_frames(frames, detections)
    assert (
        filtered_frames[:, 0, 0, 0]
        == np.array(
            [90, 80, 70, 60, 50, 87, 30, 40, 34, 10, 22, 20, 71, 16, 39, 14, 77, 65, 42, 13]
        )
    ).all()
