from pathlib import Path

from typer.testing import CliRunner
from pytest_mock import mocker  # noqa: F401

from zamba.cli import app


assets_dir = Path(__file__).parent / 'assets'
runner = CliRunner()


def test_predict_options(sample_model_path, sample_data_path, mocker):  # noqa: F811
    # before mocking predictions, test real prediction on single video using config file
    result = runner.invoke(app, ['predict', '--config', str(assets_dir / 'sample_config.yaml')])
    assert result.exit_code == 0

    # mock predictions to just test CLI args
    def pred_mock(self):
        return None

    mocker.patch('zamba.cli.ModelManager.predict', pred_mock)

    result = runner.invoke(app, ['predict', '--data-path', str(sample_data_path)])
    assert result.exit_code == 0

    result = runner.invoke(app, ['predict', '--data-path', str(sample_data_path), '--verbose'])
    assert result.exit_code == 0

    result = runner.invoke(app, ['predict', '--data-path', str(sample_data_path), '--bad-arg'])
    assert result.exit_code == 2
    assert 'no such option' in result.output

    result = runner.invoke(app, ['predict', '--data-path', str('not/a/path'), '--verbose'])
    assert result.exit_code == 2
    assert "Path 'not/a/path' does not exist" in result.output


def test_train_options(data_dir, sample_model_path, mocker):  # noqa: F811

    # before mocking ModelManager.train, check non-custom model errors
    result = runner.invoke(app, ['train', '--model-class', 'cnnensemble'])
    assert result.exit_code == 1
    assert "NotImplementedError('Currently only custom models can be trained.'" in str(result)

    # mock training to just test CLI args
    def train_mock(self):
        return None

    mocker.patch('zamba.cli.ModelManager.train', train_mock)

    result = runner.invoke(app, ['train', '--train-data', str(data_dir)])
    assert result.exit_code == 0

    result = runner.invoke(app, ['train', '--model-path', str(sample_model_path)])
    assert result.exit_code == 0

    result = runner.invoke(app, ['train', str(data_dir), '--download_region'])
    assert result.exit_code == 2
    assert 'no such option' in result.output

    result = runner.invoke(app, ['train', '--framework', 'fastai'])
    assert result.exit_code == 2
    assert "invalid choice: fastai. (choose from keras, pytorch)" in result.output

    result = runner.invoke(app, ['train', '--config', str(assets_dir / 'sample_config.yaml')])
    assert result.exit_code == 0
