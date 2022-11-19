import appdirs
import pandas as pd
from pathlib import Path
from pydantic import ValidationError
import pytest

from zamba.models.depth_estimation import DepthEstimationManager, DepthEstimationConfig

from conftest import TEST_VIDEOS_DIR


@pytest.fixture
def two_video_filepaths(tmp_path):
    output_path = tmp_path / "filepaths.csv"

    # from labels csv
    filepaths = [
        str(
            TEST_VIDEOS_DIR
            / "data/raw/goualougo_2013/gorillas_2013/MPI_FID_20_Duboscia/14-Jun-2013/FID_20_Duboscia_2013-6-14_0083.AVI"
        ),
        str(TEST_VIDEOS_DIR / "data/raw/savanna/Grumeti_Tanzania/K38_check3/09190048_Hyena.AVI"),
    ]

    pd.DataFrame(columns=["filepath"], data=filepaths).to_csv(output_path)
    return output_path


def test_prediction(two_video_filepaths):
    dem = DepthEstimationManager(model_cache_dir=Path(appdirs.user_cache_dir()) / "zamba", gpus=0)

    filepaths = pd.read_csv(two_video_filepaths).filepath.values
    preds = dem.predict(filepaths)

    assert preds.shape == (77, 3)
    assert preds.distance.notnull().sum() == 2
    assert preds.filepath.nunique() == 2

    # two detections for frame 0
    assert preds.set_index(["filepath", "time"]).loc[filepaths[0], 0].shape[0] == 2


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
