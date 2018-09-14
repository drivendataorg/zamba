import pytest
from zamba.models.cnnensemble.src import config

from zamba.models.manager import ModelManager


@pytest.mark.skip(reason="This test takes hours to run, makes network calls, and is really for local dev only.")
def test_predict():

    data_dir = config.MODEL_DIR / "input" / "raw_test"

    manager = ModelManager('', model_class='cnnensemble', proba_threshold=0.5)
    manager.predict(data_dir, save=True)
