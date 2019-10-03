import numpy as np

from zamba.models.mega_detector import MegaDetector


def test_compute_features(data_dir):
    mega = MegaDetector()
    video_paths = [path for path in data_dir.glob("*") if not path.stem.startswith(".")]
    mega_features = mega.compute_features(video_paths)

    assert mega_features.shape[1] == len(MegaDetector.FEATURE_NAMES)


def test_compute_n_detections_and_areas():
    # Compare output for test dataset example (af9b5346-1752-49dd-911a-0aeb2329ee98)
    boxes = [
        np.array([
            [0.6169399, 0.6792712, 0.95434475, 0.9707742],
            [0.5773766, 0.582577, 0.73469436, 0.71894],
        ]),
        np.array([
            [0.6552647, 0.65777284, 0.9466182, 0.8432818],
            [0.53960234, 0.23661435, 0.7348131, 0.38460845],
        ]),
        np.array([
            [0.47295967, 0.10388512, 0.8461845, 0.18570223],
        ]),
        np.array([])
    ]
    n_detections, area = MegaDetector.compute_n_detections_and_areas(boxes, h=252, w=448)
    assert n_detections == 3
    assert area == 20587

    # Parametric example
    boxes = [
        np.array(
            [
                [0.1, 0.2, 0.3, 0.5],  # area of 600
                [0.2, 0.2, 0.4, 0.7],  # only first bounding box for a frame is used
            ],
        ),
        np.array([]),
        np.array(
            [[0.55, 0.8, 0.6, 0.9]],  # area of 50
        ),
    ]
    n_detections, area = MegaDetector.compute_n_detections_and_areas(boxes, h=100, w=100)
    assert n_detections == 2
    assert area == 650
