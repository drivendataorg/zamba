from pathlib import Path
import pytest

from zamba.models.manager import ModelManager


@pytest.mark.skip(reason="Production doesn't download weights yet, and sample vids aren't baked into project")
def test_predict():

    # this test assumes a dir sitting parallel to project source
    data_dir = Path(__file__).parent.parent.parent.parent / "zamba-test-data" / "raw-vids"

    manager = ModelManager('',
                           model_class='cnnensemble',
                           proba_threshold=0.5)
    manager.predict(data_dir,
                    save=True)
