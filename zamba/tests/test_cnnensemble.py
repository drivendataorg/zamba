import pathlib
import pytest
import shutil
import tempfile

import zamba
from zamba.models.cnnensemble.src import config
from zamba.models.manager import ModelManager


@pytest.mark.skipif(
    zamba.config.codeship,
    reason="Uses too much memory for codeship build, but test locally before merging.",
)
def test_predict_fast(data_dir):
    manager = ModelManager('', model_class='cnnensemble', output_class_names=False, model_kwargs=dict(profile='fast'))
    result = manager.predict(data_dir, save=True)

    # check that duiker is most likely class (manually verified)
    assert result.idxmax(axis=1).values[0] == "duiker"

    result.to_csv(str(config.MODEL_DIR / 'output' / 'test_prediction.csv'))


@pytest.mark.skipif(
    zamba.config.codeship,
    reason="Uses too much memory for codeship build, but test locally before merging.",
)
def test_predict_fast_images(test_asset_dir):
    manager = ModelManager(model_class='cnnensemble', output_class_names=False, model_kwargs=dict(profile='fast'), )
    result = manager.predict(test_asset_dir / "test_images", save=False, predict_kwargs=dict(resample=False))

    # test assets happen to be identified as blank
    assert result.idxmax(axis=1).values[0] == "blank"


@pytest.mark.skip(reason="This test takes hours to run, makes network calls, and is really for local dev only.")
def test_predict_full(data_dir):
    manager = ModelManager('', model_class='cnnensemble', output_class_names=False, model_kwargs=dict(profile='full'))
    result = manager.predict(data_dir, save=True)
    result.to_csv(str(config.MODEL_DIR / 'output' / 'test_prediction.csv'))


@pytest.mark.skip(reason="This test takes hours to run and is really for local dev only.")
def test_train():
    manager = ModelManager(model_class='cnnensemble',
                           verbose=True,
                           model_kwargs=dict(download_weights=False))
    manager.train(config)


def test_validate_videos(data_dir):
    """Tests that all videos in the data directory are marked as valid."""
    paths = data_dir.glob("*")
    valid_videos, invalid_videos = zamba.utils.get_valid_videos(paths)
    assert len(invalid_videos) == 0


def test_load_data(data_dir):
    manager = ModelManager(
        '', model_class='cnnensemble',
        output_class_names=False,
        model_kwargs=dict(profile='fast'),
    )
    input_paths = manager.model.load_data(data_dir)
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
        '',
        model_class="cnnensemble",
        output_class_names=False,
        model_kwargs={
            "profile": "fast"
        },
    )
    predictions = manager.predict(video_directory)
    assert predictions.loc[
        predictions.index.str.contains("invalid")
    ].isnull().values.all()

    assert ~predictions.loc[
        predictions.index.str.contains("video")
    ].isnull().values.any()

    tempdir.cleanup()
