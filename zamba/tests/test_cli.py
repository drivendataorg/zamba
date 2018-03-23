from click.testing import CliRunner

from zamba.cli import predict


def test_predict_modelpath(sample_model_path, sample_data_path):

    # click's cli testing object
    runner = CliRunner()

    # test the predict cli function
    result = runner.invoke(predict, [str(sample_data_path),
                                     '--model_path', sample_model_path,
                                     '--model_class', "sample"])
    assert result.exit_code == 0
