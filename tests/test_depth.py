import appdirs
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import ValidationError
import pytest

from zamba.models.config import GPUS_AVAILABLE
from zamba.models.depth_estimation import DepthEstimationManager, DepthEstimationConfig

from conftest import ASSETS_DIR, TEST_VIDEOS_DIR


@pytest.fixture
def two_video_filepaths(tmp_path):
    output_path = tmp_path / "filepaths.csv"

    filepaths = [
        # video from depth estimation competition to verify actual preds
        str(ASSETS_DIR / "depth_tests" / "aava.mp4"),
        # test asset video with no detections
        str(TEST_VIDEOS_DIR / "data/raw/savanna/Grumeti_Tanzania/K38_check3/09190048_Hyena.AVI"),
    ]

    pd.DataFrame(columns=["filepath"], data=filepaths).to_csv(output_path)
    return output_path


def test_prediction(two_video_filepaths):
    dem = DepthEstimationManager(
        model_cache_dir=Path(appdirs.user_cache_dir()) / "zamba", gpus=GPUS_AVAILABLE
    )

    filepaths = pd.read_csv(two_video_filepaths).filepath.values
    preds = dem.predict(filepaths)

    # NB: we expect some small differences in number of detections across operating systems
    assert len(preds) >= 80
    assert preds.distance.notnull().sum() >= 40
    assert preds.filepath.nunique() == 2

    # predictions for reference video
    ref_vid_preds = preds[preds.filepath == filepaths[0]].set_index("time")

    # two animals found at time 30
    assert len(ref_vid_preds.loc[30]) == 2

    # confirm distance values
    assert np.isclose(
        ref_vid_preds.loc[[30, 40, 50]].distance.values,
        [3.1, 3.1, 3.6, 3.6, 4.1],
    ).all()

    # check nan rows exist for video with no detection
    no_det_preds = preds[preds.filepath == filepaths[1]].set_index("time")
    assert len(no_det_preds) == 15
    assert no_det_preds.distance.isnull().all()


def test_duplicate_filepaths_are_ignored(tmp_path, two_video_filepaths):
    # create duplicate file that gets skipped
    df = pd.read_csv(two_video_filepaths)
    double_df = pd.concat([df, df])
    assert len(double_df) == len(df) * 2

    config = DepthEstimationConfig(filepaths=double_df, save_to=tmp_path)
    assert len(config.filepaths) == len(df)


def test_save_dir_and_overwrite(tmp_path, two_video_filepaths):
    # create empty pred file to force use of overwrite
    preds_path = tmp_path / "depth_estimation.csv"
    preds_path.touch()

    with pytest.raises(ValidationError):
        DepthEstimationConfig(filepaths=two_video_filepaths, save_to=preds_path)

    # this works if overwrite is passed
    config = DepthEstimationConfig(
        filepaths=two_video_filepaths, save_to=preds_path, overwrite=True
    )
    assert config.overwrite


def test_invalid_video_is_skipped(tmp_path):
    # create invalid vid
    invalid_video = tmp_path / "invalid_vid.mp4"
    invalid_video.touch()

    config = DepthEstimationConfig(
        filepaths=pd.DataFrame(columns=["filepath"], data=[invalid_video]),
        save_to=tmp_path / "preds.csv",
    )
    config.run_model()

    # ensure outputs get written out and but is empty since video could not be loaded
    preds = pd.read_csv(tmp_path / "preds.csv")
    assert len(preds) == 0
    assert (preds.columns == ["filepath", "time", "distance"]).all()
