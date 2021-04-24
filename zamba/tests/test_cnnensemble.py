import pathlib
import pytest
import shutil
import tempfile

import zamba
from zamba.models.cnnensemble.src import config
from zamba.models.cnnensemble_model import CnnEnsemble
from zamba.models.manager import ModelManager, PredictConfig, ModelConfig


@pytest.mark.skipif(
    zamba.config.codeship,
    reason="Uses too much memory for codeship build, but test locally before merging.",
)
def test_predict_fast(data_dir):
    manager = ModelManager(
        model_config=ModelConfig(
            model_class='cnnensemble',
            model_kwargs=dict(profile='fast'),
        ),
        predict_config=PredictConfig(
            output_class_names=False,
            data_path=data_dir,
            save=True,
            pred_path=str(config.MODEL_DIR / 'output' / 'test_prediction.csv'),
        )
    )
    result = manager.predict()

    # check that duiker is most likely class (manually verified)
    assert result.idxmax(axis=1).values[0] == "duiker"


@pytest.mark.skip(reason="This test takes hours to run, makes network calls, and is really for local dev only.")
def test_predict_full(data_dir):
    manager = ModelManager(
        model_config=ModelConfig(
            model_class='cnnensemble',
            model_kwargs=dict(profile='full'),
        ),
        predict_config=PredictConfig(
            output_class_names=False,
            data_path=data_dir,
            save=True,
            pred_path=str(config.MODEL_DIR / 'output' / 'test_prediction.csv'),
        )
    )
    manager.predict()


def test_validate_videos(data_dir):
    """Tests that all videos in the data directory are marked as valid."""
    paths = data_dir.glob("*")
    valid_videos, invalid_videos = zamba.utils.get_valid_videos(paths)
    assert len(invalid_videos) == 0


def test_load_data(data_dir):
    model = CnnEnsemble(profile="fast")
    input_paths = model.load_data(data_dir)
    assert len(input_paths) > 0


@pytest.mark.skipif(
    zamba.config.codeship,
    reason="Uses too much memory for codeship build, but test locally before merging.",
)
def test_predict_invalid_videos(data_dir):
    """Tests whether invalid videos are correctly skipped."""
    tempdir = tempfile.TemporaryDirectory()
    video_directory = pathlib.Path(tempdir.name)

    # create invalid (empty) videos
    for i in range(2):
        (video_directory / f"invalid{i:02}.mp4").touch()

    # copy valid videos
    test_video_path = list(data_dir.glob("*.mp4"))[0]
    for i in range(2):
        shutil.copy(test_video_path, video_directory / f"video{i:02}.mp4")

    manager = ModelManager(
        model_config=ModelConfig(
            model_class="cnnensemble",
            model_kwargs={
                "profile": "fast"
            },
        ),
        predict_config=PredictConfig(
            output_class_names=False,
            data_path=video_directory,
        )
    )
    predictions = manager.predict()
    assert predictions.loc[
        predictions.index.str.contains("invalid")
    ].isnull().values.all()

    assert ~predictions.loc[
        predictions.index.str.contains("video")
    ].isnull().values.any()

    tempdir.cleanup()
