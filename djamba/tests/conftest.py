from pathlib import Path
import pickle
from shutil import rmtree

import numpy as np
import pytest

from djamba.models.model import SampleModel


@pytest.fixture
def model_path():
    project_src = Path(__file__).absolute().parent.parent
    model_dir = project_src / "models" / "assets"

    model_name = Path("test-model.h5")
    model_subdir = model_dir / model_name.stem
    model_subdir.mkdir(exist_ok=True)

    return model_subdir / model_name


@pytest.fixture
def sample_model_path():
    model = SampleModel()
    path = model_path()
    model.save_model(path=path)
    yield path
    rmtree(path.parent)


@pytest.fixture
def sample_data_path():
    sample_data = [np.array([6, 0.3]),
                   np.array([3, 0.1])]

    project_src = Path(__file__).absolute().parent.parent
    data_path = project_src / "tests" / "data" / "sample_data.pkl"
    data_path.parent.mkdir(exist_ok=True)
    with open(data_path, 'wb') as f:
        pickle.dump(sample_data, f)

    assert data_path.resolve().exists()
    yield data_path
    rmtree(data_path.parent)
