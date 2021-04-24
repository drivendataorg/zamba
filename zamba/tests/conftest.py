from pathlib import Path
import pickle
from shutil import rmtree

import numpy as np
import pytest

from zamba.tests.sample_model import SampleModel
from zamba.models.cnnensemble.src import config


@pytest.fixture
def sample_model_path():
    """This fixture creates a sample model, saves it to the sample path,
    then yields the path to the sample.

    Removes model once test is complete or fails.

    Returns (Path): path to test model

    """
    model_path = Path(__file__).parents[1] / "models" / "assets" / "test_model.h5"

    model = SampleModel()
    model.to_disk(path=model_path)

    yield model_path
    rmtree(model_path.parent)


@pytest.fixture
def sample_data_path():
    """This fixture creates sample data, saves it, yields path to load it.

    Removes the data once test is complete or if test fails.

    Returns:

    """
    sample_data = [np.array([6, 0.3]),
                   np.array([3, 0.1])]

    data_path = Path(__file__).parent / "data" / "sample_data.pkl"
    data_path.parent.mkdir(exist_ok=True)
    with open(data_path, 'wb') as f:
        pickle.dump(sample_data, f)

    assert data_path.resolve().exists()
    yield data_path
    rmtree(data_path.parent)


@pytest.fixture
def data_dir():
    return config.MODEL_DIR / "input" / "raw_test"
