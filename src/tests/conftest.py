from pathlib import Path
import pytest

@pytest.fixture
def project_src():
    return Path(__file__).absolute().parent.parent

@pytest.fixture
def models_dir():
    return project_src() / "src" / "models" / "assets"

@pytest.fixture
def model_name():
    return "test-model"

@pytest.fixture
def model_subdir():
    subdir = Path(models_dir(), model_name())
    subdir.mkdir(exist_ok=True)
    return subdir

@pytest.fixture
def global_step():
    """
    Store global step,
    which will be baked into metagraph file name
    """
    return 1000

@pytest.fixture
def input_names():
    """
    Store names of input layer tensors.
    For the test model, we simply input w1, w2,
    add them, and multiply by bias:
    (w1 + w2) * b1
    """
    return ["w1", "w2", "bias"]

@pytest.fixture
def op_to_restore_name():
    """operation (eg predict) to restore"""
    return "op_to_restore"