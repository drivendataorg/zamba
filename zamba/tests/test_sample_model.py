import numpy as np

from zamba.models.manager import ModelManager


def test_create_and_save(sample_model_path, sample_data_path):

    manager = ModelManager(sample_model_path, model_class='sample')

    # "predict" (add, multiply), return exact values since no thresh given
    result = manager.predict(sample_data_path)

    # 6 + 3 == 9
    assert result.iloc[0].added == 9

    # 6 * 3 == 18
    assert result.iloc[0].multiplied == 18

    # 0.3 + 0.1 == 0.4
    assert result.iloc[1].added == np.float32(0.3) + np.float32(0.1)

    # 0.3 * 0.1 == 0.03
    assert result.iloc[1].multiplied == np.float32(0.3) * np.float32(0.1)

    manager.model.save_model()
    assert manager.model_path.exists()


def test_load_and_predict(sample_model_path, sample_data_path):

    # load the sample model in the ModelManager
    manager = ModelManager(sample_model_path,
                           model_class='sample',
                           proba_threshold=0.5)

    # "predict" (add, multiply), return binary since thresh given
    preds = manager.predict(sample_data_path)

    # 6 + 3 == 9 >= 0.5 --> True
    assert preds.iloc[0].added

    # 6 * 3 == 18 >= 0.5 --> True
    assert preds.iloc[0].multiplied

    # 0.3 + 0.1 == 0.4 <= 0.5 --> False
    assert not preds.iloc[1].added

    # 0.3 * 0.1 == 0.03 <= 0.5 --> False
    assert not preds.iloc[1].multiplied
