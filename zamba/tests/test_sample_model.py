from pathlib import Path

import numpy as np

from zamba.tests.sample_model import SampleModel
from zamba.models.manager import ModelManager, PredictConfig, ModelConfig


def test_load_and_save():
    sm = SampleModel()
    save_path = Path("my_model.h5")
    sm.save_model(save_path)
    assert save_path.exists()

    assert SampleModel.load(model_path=save_path)
    save_path.unlink()


def test_load_and_predict(sample_model_path, sample_data_path):
    manager = ModelManager(
        model_config=ModelConfig(
            model_class="sample",
            model_kwargs=dict()
        ),
        predict_config=PredictConfig(
            data_path=sample_data_path,
        )
    )

    # "predict" (add, multiply), return exact values since no thresh given
    result = manager.predict()

    # 6 + 3 == 9
    assert result.iloc[0].added == 9

    # 6 * 3 == 18
    assert result.iloc[0].multiplied == 18

    # 0.3 + 0.1 == 0.4
    assert result.iloc[1].added == np.float32(0.3) + np.float32(0.1)

    # 0.3 * 0.1 == 0.03
    assert result.iloc[1].multiplied == np.float32(0.3) * np.float32(0.1)


def test_load_and_predict_threshold(sample_model_path, sample_data_path):

    # load the sample model in the ModelManager
    manager = ModelManager(
        model_config=ModelConfig(
            model_class="sample",
            model_kwargs=dict()
        ),
        predict_config=PredictConfig(
            data_path=sample_data_path,
            proba_threshold=0.5,
        )
    )

    # "predict" (add, multiply), return binary since thresh given
    preds = manager.predict()

    # 6 + 3 == 9 >= 0.5 --> True
    assert preds.iloc[0].added

    # 6 * 3 == 18 >= 0.5 --> True
    assert preds.iloc[0].multiplied

    # 0.3 + 0.1 == 0.4 <= 0.5 --> False
    assert not preds.iloc[1].added

    # 0.3 * 0.1 == 0.03 <= 0.5 --> False
    assert not preds.iloc[1].multiplied
