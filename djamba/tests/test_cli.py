from click.testing import CliRunner

from djamba.cli import predict


def test_predict_modelpath(sample_model_path, sample_data_path):
    """This needs work"""

    assert sample_model_path.resolve().exists()
    assert sample_data_path.resolve().exists()

    runner = CliRunner()
    result = runner.invoke(predict, [str(sample_data_path),
                                     '--model_path', sample_model_path,
                                     '--model_class', "sample"])
    assert result.exit_code == 0
