from shutil import rmtree

from click.testing import CliRunner

from src.cli import predict
from src.src.models.io import save_model
from src.src.models.model import SampleModel


def test_predict_modelpath(model_path, mocker):
    """This needs work"""

    # instantiate new model
    model = SampleModel()

    # save model
    save_model(model, model_path)

    mocker.patch.object(SampleModel, 'predict')
    SampleModel.predict.return_value = 1

    runner = CliRunner()
    result = runner.invoke(predict, ['--modelpath', model_path,
                                     '--sample_model', True])
    assert result.exit_code == 0

    # delete test model
    rmtree(model_path.parent)
