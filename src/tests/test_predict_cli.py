from pathlib import Path
import click
from click.testing import CliRunner

import pandas as pd

from src.cli import predict

# from src.tests.test_model_config import *


def test_predict_modelpath(model_subdir):
    """
    This needs work.
    Saving the below for refrence later,
    in case we deicide to test cli directly.
    """

    # runner = CliRunner()
    # modelpath = str(model_subdir.resolve())
    # assert type(modelpath) == str
    # modelpath = str(Path('..', 'models', 'assets').resolve())
    # result = runner.invoke(predict, ['--modelpath', modelpath])
    # assert result.exit_code == 0

    pass

