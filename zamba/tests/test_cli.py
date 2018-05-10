from click.testing import CliRunner
from pytest_mock import mocker

from zamba.cli import predict


def test_predict_modelpath(sample_model_path, sample_data_path, mocker):
    # mock predictions to just test CLI args
    def pred_mock(self, ata_path, pred_path='', save=False):
        return None

    mocker.patch('zamba.cli.ModelManager.predict', pred_mock)

    # click's cli testing object
    runner = CliRunner()

    # test the predict cli function
    result = runner.invoke(predict, [str(sample_data_path)])
    assert result.exit_code == 0

    result = runner.invoke(predict, [str(sample_data_path),
                                     '--verbose'])
    assert result.exit_code == 0

    result = runner.invoke(predict, [str(sample_data_path),
                                     '--bad-arg'])
    assert result.exit_code == 2
    assert 'no such option' in result.output

    result = runner.invoke(predict, [str('not/a/path'),
                                     '--verbose'])
    assert result.exit_code == 2
    assert 'Path "not/a/path" does not exist' in result.output
