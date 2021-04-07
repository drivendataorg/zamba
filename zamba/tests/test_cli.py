from typer.testing import CliRunner
from pytest_mock import mocker  # noqa: F401

from zamba.cli import app

runner = CliRunner()


def test_predict_modelpath(sample_model_path, sample_data_path, mocker):  # noqa: F811
    # mock predictions to just test CLI args
    def pred_mock(self):
        return None

    mocker.patch('zamba.cli.ModelManager.predict', pred_mock)

    # test the predict cli function
    result = runner.invoke(app, ['predict', str(sample_data_path)])
    assert result.exit_code == 0

    result = runner.invoke(app, ['predict', str(sample_data_path), '--verbose'])
    assert result.exit_code == 0

    result = runner.invoke(app, ['predict', str(sample_data_path), '--bad-arg'])
    assert result.exit_code == 2
    assert 'no such option' in result.output

    result = runner.invoke(app, ['predict', str('not/a/path'), '--verbose'])
    assert result.exit_code == 2
    assert "Path 'not/a/path' does not exist" in result.output
