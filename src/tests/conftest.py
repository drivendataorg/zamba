from pathlib import Path
from shutil import rmtree

import pytest

from src.src.models.model import SampleModel


@pytest.fixture
def model_path():
    project_src = Path(__file__).absolute().parent.parent
    model_dir = project_src / "src" / "models" / "assets"

    model_name = Path("test-model.h5")
    model_subdir = model_dir / model_name.stem
    model_subdir.mkdir(exist_ok=True)

    return model_subdir / model_name


@pytest.fixture
def sample_model():
    model = SampleModel()
    yield model
    rmtree(model.modeldir.parent)
