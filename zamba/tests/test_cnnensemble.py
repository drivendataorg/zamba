from zamba.models.manager import ModelManager


def test_predict():
    manager = ModelManager('', model_class='cnnensemble')

    manager.predict('')
