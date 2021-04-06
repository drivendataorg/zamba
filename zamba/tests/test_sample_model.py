import numpy as np

from zamba.models.manager import ModelManager, PredictConfig


def test_create_and_save(sample_model_path, sample_data_path):

    # test with default params
    manager = ModelManager(predict_config=PredictConfig(model_class='sample'))

    # use sample model fixture
    manager = ModelManager(predict_config=PredictConfig(
        model_path=sample_model_path,
        model_class='sample',
        data_path=sample_data_path)
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

    manager.model.save_model()
    assert manager.model.model_path.exists()


def test_load_and_predict(sample_model_path, sample_data_path):

    # load the sample model in the ModelManager
    manager = ModelManager(predict_config=PredictConfig(
        model_path=sample_model_path,
        data_path=sample_data_path,
        model_class='sample',
        proba_threshold=0.5)
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
