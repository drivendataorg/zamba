from click.testing import CliRunner

from djamba.cli import predict
from djamba.models.manager import ModelManager


def test_predict_modelpath(sample_model_path, mocker):
    """This needs work"""

    # configure mocker
    mocker.patch.object(ModelManager, 'predict')
    ModelManager.predict.return_value = 1

    runner = CliRunner()
    result = runner.invoke(predict, ['--modelpath', sample_model_path,
                                     '--model_class', "sample"])
    assert result.exit_code == 0
