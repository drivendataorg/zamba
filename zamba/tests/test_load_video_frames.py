import pytest
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel, ValidationError

from zamba_algorithms.data.video import load_video_frames, VideoLoaderConfig
from zamba_algorithms.settings import ROOT_DIRECTORY

TEST_DATA_DIRECTORY = ROOT_DIRECTORY / "tests" / "assets" / "videos"


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
        (video_shape["height"] == kwargs["video_height"])
        and (video_shape["width"] == kwargs["video_width"])
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
            "video_height": 10,
            "video_width": 10,
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
            "video_height": 10,
            "video_width": 10,
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
        parameters={"scene_threshold": 0.05, "video_height": 10, "video_width": 10},
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
        parameters={"video_height": 10, "video_width": 10},
        expected_output={"height": 10, "width": 10},
    ),
    Case(
        name="video_height_width_and_total_frames",
        parameters={"video_height": 10, "video_width": 10, "total_frames": 10},
        expected_output={"height": 10, "width": 10, "frames": 10},
    ),
    Case(
        name="video_height_width_and_fps",
        parameters={"video_height": 10, "video_width": 10, "fps": 10},
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
        parameters={"megadetector_lite": 0.25, "early_bias": True},
        validation_error=True,
    ),
    Case(
        name="megadetector_and_evenly_sample",
        parameters={
            "megadetector_lite": 0.25,
            "total_frames": 10,
            "evenly_sample_total_frames": True,
        },
        validation_error=True,
    ),
    Case(
        name="megadetector_and_two_total_frames",
        parameters={
            "megadetector_lite": 0.01,
            "total_frames": 2,
            "ensure_total_frames": False,
            "fps": 2,
        },
        expected_output={"frames": assert_megadetector_total_or_none},
    ),
    Case(
        name="megadetector_and_video_height_width",
        parameters={
            "megadetector_lite": 0.01,
            "video_height": 50,
            "video_width": 50,
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
    test_video_paths = [path for path in TEST_DATA_DIRECTORY.rglob("*") if path.is_file()]
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
    params=video_metadata_values, ids=[metadata["path"].stem for metadata in video_metadata_values]
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


def test_same_filename_new_kwargs():
    """Test that load_video_frames does not load the npz file if the params change.
    """
    # use first test video
    test_vid = [f for f in TEST_DATA_DIRECTORY.rglob("*") if f.is_file()][0]

    first_load = load_video_frames(test_vid, fps=1)
    new_params_same_name = load_video_frames(test_vid, fps=2)
    assert first_load != new_params_same_name

    # check no params
    first_load = load_video_frames(test_vid)
    assert first_load != new_params_same_name

    # multiple params in config
    c1 = VideoLoaderConfig(scene_threshold=0.2)
    c2 = VideoLoaderConfig(scene_threshold=0.2, crop_bottom_pixels=2)

    first_load = load_video_frames(filepath=test_vid, config=c1)
    new_params_same_name = load_video_frames(filepath=test_vid, config=c2)
    assert first_load != new_params_same_name
