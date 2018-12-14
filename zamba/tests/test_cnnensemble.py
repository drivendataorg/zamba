import pytest
from zamba.models.cnnensemble.src import config
from zamba.models.manager import ModelManager


@pytest.mark.skip(reason="This test takes hours to run, makes network calls, and is really for local dev only.")
def test_predict():

    data_dir = config.MODEL_DIR / "input" / "raw_test"

    manager = ModelManager('', model_class='cnnensemble', output_class_names=True)
    result = manager.predict(data_dir, save=True)
    result.to_csv(str(config.MODEL_DIR / 'output' / 'test_prediction.csv'))


@pytest.mark.skip(reason="This test takes hours to run and is really for local dev only.")
def test_train():
    manager = ModelManager(model_class='cnnensemble',
                           model_kwargs=dict(download_weights=False, verbose=True))
    manager.train(config)
