import json

import numpy as np
from PIL import Image
import pytest
import torch

from zamba.object_detection import YoloXModel, YoloXExp, YoloXArgs
from zamba.object_detection.yolox.megadetector_lite_yolox import (
    MegadetectorLiteYoloX,
    MegadetectorLiteYoloXConfig,
)

from conftest import ASSETS_DIR


@pytest.fixture
def dog():
    return Image.open(ASSETS_DIR / "dog.jpg")


@pytest.fixture
def dummy_yolox_path(tmp_path):
    yolox = YoloXModel(exp=YoloXExp(num_classes=1), args=YoloXArgs())
    checkpoint = {"model": yolox.exp.get_model().state_dict()}
    torch.save(checkpoint, tmp_path / "dummy_yolox.pth")
    return tmp_path / "dummy_yolox.pth"


@pytest.fixture
def dummy_yolox_model_kwargs(tmp_path):
    kwargs = dict(num_classes=1, image_size=640, backbone="yolox-tiny")
    json_path = tmp_path / "dummy_yolox_kwargs.json"
    with json_path.open("w+") as f:
        json.dump(kwargs, f)
    return json_path


def test_load_megadetector(dummy_yolox_path, dummy_yolox_model_kwargs):
    MegadetectorLiteYoloX(
        dummy_yolox_path, dummy_yolox_model_kwargs, MegadetectorLiteYoloXConfig()
    )


def test_scale_and_pad_array():
    original_array = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.uint8)
    output_array = MegadetectorLiteYoloX.scale_and_pad_array(
        original_array, output_width=3, output_height=4
    )
    assert output_array.shape == (4, 3, 3)
    assert (output_array[:3] == original_array).all()
    assert (output_array[3] == np.zeros(3)).all()

    output_array = MegadetectorLiteYoloX.scale_and_pad_array(
        original_array, output_width=4, output_height=3
    )
    assert output_array.shape == (3, 4, 3)
    assert (output_array[:, :3] == original_array).all()
    assert (output_array[:, 3] == np.zeros(3)).all()


def test_detect_image(mdlite, dog):
    mdlite = MegadetectorLiteYoloX()
    boxes, scores = mdlite.detect_image(np.array(dog))

    assert len(scores) == 1
    assert np.allclose([0.65678996, 0.21596366, 0.71104807, 0.277931], boxes[0])


def test_detect_video(mdlite, dog):
    total_frames = 10
    object_frame_indices = [0, 3, 4, 6]

    video = np.zeros([total_frames] + list(dog.size[::-1]) + [3], dtype=np.uint8)
    video[object_frame_indices] = np.array(dog)

    mdlite = MegadetectorLiteYoloX()

    detections = mdlite.detect_video(video)

    # Check that we detected the correct number of frames
    assert sum(len(score) > 0 for _, score in detections) == len(object_frame_indices)

    for frame_index, (frame, (_, score)) in enumerate(zip(video, detections)):
        if len(score) > 0:
            # Frame index is in intended frame indices
            assert frame_index in object_frame_indices

            # No blank frames were selected
            assert not (frame == np.zeros(frame.shape, dtype=np.uint8)).all()
