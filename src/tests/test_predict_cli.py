import click
from click.testing import CliRunner

import pandas as pd

from src.cli import predict

from src.tests import test_model_config as test_config

modeldir = test_config.model_subdir

def test_predict_modelpath():
    """This needs work"""

    runner = CliRunner()
    result = runner.invoke(predict, ['--modelpath', modeldir])
    assert result.exit_code == 0

