import numpy as np

from djamba.models.manager import ModelManager


def test_create_and_save(sample_model_path):

    # create some sample data
    data = [np.array([4, 5]),
            np.array([8, 9])]

    manager = ModelManager(sample_model_path, model_class='sample')

    # "predict" (add, multiply), return exact values since no thresh given
    result = manager.predict(data)

    # 4 + 8 == 12
    assert result.iloc[0].added == 12

    # 4 * 8 == 32
    assert result.iloc[0].multiplied == 32

    # 5 + 9 == 14
    assert result.iloc[1].added == 14

    # 5 * 9 == 45
    assert result.iloc[1].multiplied == 45

    manager.model.save_model()


def test_load_and_predict(sample_model_path):
    """
    Simple load of Model object using graph
    in test_model_save_and_load to predict
    and save out
    """

    # load the sample model in the ModelManager
    model_manager = ModelManager(sample_model_path,
                                 model_class='sample',
                                 proba_thresh=0.5)

    # sample data
    new_data = [np.array([6, 0.3]),
                np.array([3, 0.1])]

    # # "predict" (add, multiply), return binary since thresh given
    preds = model_manager.predict(new_data)

    # 6 + 3 == 9 >= 0.5 --> True
    assert preds.iloc[0].added

    # 6 * 3 == 18 >= 0.5 --> True
    assert preds.iloc[0].multiplied

    # 0.3 + 0.1 == 0.4 <= 0.5 --> False
    assert not preds.iloc[1].added

    # 0.3 * 0.1 == 0.03 <= 0.5 --> False
    assert not preds.iloc[1].multiplied
