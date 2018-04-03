from pathlib import Path
import pytest

from zamba.models.manager import ModelManager


@pytest.mark.skip(reason="This test takes hours to run, makes network calls, and is really for local dev only.")
def test_predict():

    data_dir = Path(__file__).parent.parent / "models" / "cnnensemble" / "input" / "raw_test"

    manager = ModelManager('', model_class='cnnensemble', proba_threshold=0.5)
    manager.predict(data_dir, save=True)
