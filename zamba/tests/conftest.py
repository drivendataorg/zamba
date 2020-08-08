from pathlib import Path
import pickle
from shutil import rmtree

import numpy as np
import pytest

from zamba.models.model import SampleModel
from zamba.models.cnnensemble.src import config


@pytest.fixture
def model_path():
    """This fixture creates a path to and filename for a test model.

    Returns (Path): path to sample model

    """
    project_src = Path(__file__).absolute().parent.parent
    model_dir = project_src / "models" / "assets"

    model_name = Path("test-model.h5")
    model_subdir = model_dir / model_name.stem
    model_subdir.mkdir(exist_ok=True)

    return model_subdir / model_name


@pytest.fixture
def sample_model_path(model_path):
    """This fixture creates a sample model, saves it to the sample path,
    then yields the path to the sample.

    Removes model once test is complete or fails.

    Returns (Path): path to test model

    """
    model = SampleModel()
    model.save_model(path=model_path)
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
