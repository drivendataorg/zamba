import pytest

from zamba.models.manager import ModelManager


@pytest.mark.xfail(reason="Not yet implemented!")
def test_predict(mocker):
    manager = ModelManager('', model_class='cnnensemble')
    manager.predict('')
