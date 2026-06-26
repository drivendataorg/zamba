import numpy as np
import pytest

from zamba.object_detection.yolox.megadetector_lite_yolox import MegadetectorLiteYoloXConfig

pytestmark = pytest.mark.video

n_frames = 100


@pytest.fixture
def frames():
    rng = np.random.RandomState(68891)
    # 20 6x8 RGB frames where the upper-right hand pixel is the frame index
    frames = rng.randint(0, 255, size=(n_frames, 6, 8, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(n_frames, dtype=np.uint8)
    return frames


@pytest.fixture
def detections():
    # Keep detection generation independent from the frames fixture so weighted samples do not
    # depend on fixture evaluation order.
    rng = np.random.RandomState(68891)
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


def test_score_sorted_tie_break_is_by_frame_index(mdlite, frames):
    """Zero-score ties must pick lowest frame indices, not pandas-version-dependent order."""
    detections = [
        (
            np.array([]),
            np.array([0.9 if i in (5, 10) else 0.0]),  # only frames 5, 10 detected
        )
        for i in range(20)
    ]
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=0.25,
        n_frames=6,
        fill_mode="score_sorted",
        sort_by_time=False,
    )
    out = mdlite.filter_frames(frames[:20], detections)
    # top 2 by score: 5, 10; fill remaining 4 zeros with the lowest frame indices: 0, 1, 2, 3
    assert list(out[:, 0, 0, 0]) == [5, 10, 0, 1, 2, 3]


def test_above_threshold_tie_break_is_by_frame_index(mdlite, frames):
    """When more than n_frames share the top score, lowest frame indices win."""
    detections = [
        (np.array([]), np.array([0.9])) for _ in range(10)
    ]
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=0.25,
        n_frames=5,
        fill_mode="score_sorted",
        sort_by_time=False,
    )
    out = mdlite.filter_frames(frames[:10], detections)
    assert list(out[:, 0, 0, 0]) == [0, 1, 2, 3, 4]


def test_n_frames_none_returns_all_above_threshold(mdlite, frames):
    detections = [
        (np.array([]), np.array([0.9 if i in (2, 7) else 0.0])) for i in range(10)
    ]
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=0.25,
        n_frames=None,
        fill_mode="score_sorted",
        sort_by_time=False,
    )
    out = mdlite.filter_frames(frames[:10], detections)
    assert list(out[:, 0, 0, 0]) == [2, 7]


def test_all_zero_scores_fills_by_lowest_frame_index(mdlite, frames):
    detections = [(np.array([]), np.array([0.0])) for _ in range(10)]
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=0.25,
        n_frames=5,
        fill_mode="score_sorted",
        sort_by_time=False,
    )
    out = mdlite.filter_frames(frames[:10], detections)
    assert list(out[:, 0, 0, 0]) == [0, 1, 2, 3, 4]


def test_filter_frames_repeated_runs_identical(mdlite, frames):
    detections = [
        (np.array([]), np.array([0.9 if i in (5, 10) else 0.0])) for i in range(20)
    ]
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=0.25,
        n_frames=6,
        fill_mode="score_sorted",
        sort_by_time=False,
    )
    first = mdlite.filter_frames(frames[:20], detections)
    second = mdlite.filter_frames(frames[:20], detections)
    np.testing.assert_array_equal(first, second)


def test_empty_detections_returns_empty(mdlite, frames):
    mdlite.config = MegadetectorLiteYoloXConfig(
        confidence=0.25,
        n_frames=5,
        fill_mode="score_sorted",
        sort_by_time=False,
    )
    out = mdlite.filter_frames(frames[:0], [])
    assert out.shape[0] == 0


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
            [90, 80, 70, 60, 72, 96, 66, 75, 95, 93, 69, 23, 76, 57, 45, 83, 50, 51, 56, 73]
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
            [90, 80, 70, 60, 50, 13, 30, 40, 49, 10, 34, 14, 52, 36, 81, 99, 79, 24, 0, 20]
        )
    ).all()
