import shutil

from typer.testing import CliRunner
import pandas as pd
import pytest
from pytest_mock import mocker  # noqa: F401

from zamba.cli import app

from conftest import ASSETS_DIR, TEST_VIDEOS_DIR

runner = CliRunner()


@pytest.fixture
def minimum_valid_train(labels_absolute_path):
    return ["train", "--labels", str(labels_absolute_path), "--skip-load-validation"]


@pytest.fixture
def minimum_valid_predict():
    return ["predict", "--data-dir", str(TEST_VIDEOS_DIR), "--skip-load-validation"]


# mock training to just test CLI args
def train_mock(self):
    return None


# mock predictions to just test CLI args
def pred_mock(self):
    return None


def test_train_specific_options(mocker, minimum_valid_train, tmp_path):  # noqa: F811
    mocker.patch("zamba.cli.ModelManager.train", train_mock)

    # check labels must exist
    result = runner.invoke(app, ["train", "--labels", "my_labels.csv"])
    assert result.exit_code == 2
    assert "'my_labels.csv' does not exist." in result.output

    # check data dir must exist
    result = runner.invoke(app, ["train", "--data-dir", "my_data"])
    assert result.exit_code == 2
    assert "Path 'my_data' does not exist." in result.output

    # test from config
    result = runner.invoke(
        app, ["train", "--config", str(ASSETS_DIR / "sample_train_config.yaml")]
    )
    assert result.exit_code == 0
    assert f"Config file: {str(ASSETS_DIR / 'sample_train_config.yaml')}" in result.output

    result = runner.invoke(app, minimum_valid_train + ["--save-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_shared_cli_options(mocker, minimum_valid_train, minimum_valid_predict):  # noqa: F811
    """Test CLI options that are shared between train and predict commands."""

    mocker.patch("zamba.cli.ModelManager.train", train_mock)
    mocker.patch("zamba.cli.ModelManager.predict", pred_mock)

    for command in [minimum_valid_train, minimum_valid_predict]:

        # check default model is time distributed one
        result = runner.invoke(app, command)
        assert result.exit_code == 0
        assert "Config file: None" in result.output

        # check all models options are valid
        for model in ["time_distributed", "slowfast", "european"]:
            result = runner.invoke(app, command + ["--model", model])
            assert result.exit_code == 0

        # check invalid model name raises error
        result = runner.invoke(app, command + ["--model", "my_model"])
        assert result.exit_code == 2
        assert "Invalid value" in result.output

        # test batch size, gpus, num_workers, and dry run options
        result = runner.invoke(
            app,
            command
            + [
                "--batch-size",
                "10",
                "--gpus",
                "0",
                "--num-workers",
                "1",
                "--dry-run",
                "-y",
            ],
        )
        assert result.exit_code == 0

        # invalid extra arg
        result = runner.invoke(app, command + ["--bad-arg"])
        assert result.exit_code == 2
        assert "no such option" in str(result.output).lower()

        # validation error with too many gpus
        result = runner.invoke(app, command + ["--gpus", "2"])
        assert result.exit_code == 1
        assert "Cannot use 2" in str(result.exc_info)


def test_predict_specific_options(mocker, minimum_valid_predict, tmp_path):  # noqa: F811
    mocker.patch("zamba.cli.ModelManager.predict", pred_mock)

    # check data dir must exist
    result = runner.invoke(app, ["predict", "--data-dir", "my_data"])
    assert result.exit_code == 2
    assert "Path 'my_data' does not exist." in result.output

    # check checkpoint must exist
    result = runner.invoke(
        app,
        minimum_valid_predict + ["--checkpoint", "my_checkpoint.ckpt"],
    )
    assert result.exit_code == 2
    assert "Path 'my_checkpoint.ckpt' does not exist." in result.output

    # test prob_threshold invalid option
    result = runner.invoke(app, minimum_valid_predict + ["--proba-threshold", 5])
    assert result.exit_code == 1
    assert (
        "Setting proba_threshold outside of the range (0, 1) will cause all probabilities to be rounded to the same value"
        in str(result.exc_info)
    )

    # test valid output args
    result = runner.invoke(
        app,
        minimum_valid_predict + ["--proba-threshold", "0.5", "--no-save"],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        minimum_valid_predict + ["--output-class-names", "--save"],
    )
    assert result.exit_code == 0


def test_actual_prediction_on_single_video(tmp_path):  # noqa: F811
    data_dir = tmp_path / "videos"
    data_dir.mkdir()
    shutil.copy(TEST_VIDEOS_DIR / "data" / "raw" / "benjamin" / "04250002.MP4", data_dir)

    save_path = tmp_path / "zamba" / "my_preds.csv"

    # Prior to mocking, run one real prediction using config
    result = runner.invoke(
        app,
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--config",
            str(ASSETS_DIR / "sample_predict_config.yaml"),
            "--yes",
            "--save-path",
            str(save_path),
        ],
    )
    assert result.exit_code == 0
    # check preds file got saved out
    assert save_path.exists()
    # check config got saved out too
    assert (save_path.parent / "predict_configuration.yaml").exists()
    assert pd.read_csv(save_path, index_col="filepath").idxmax(axis=1).values[0] == "monkey_prosimian"
