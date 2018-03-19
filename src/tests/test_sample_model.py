from shutil import rmtree

import numpy as np

from src.src.models.io import save_model, load_model
from src.src.models.model import SampleModel


def test_create_and_save(model_path):

    # instantiate new model
    model = SampleModel()

    # create some sample data
    data = [np.array([4, 5]),
            np.array([8, 9])]

    # "predict" (add, multiply), return exact values since no thresh given
    result = model.predict(data)

    # 4 + 8 == 12
    assert result.iloc[0].added == 12

    # 4 * 8 == 32
    assert result.iloc[0].multiplied == 32

    # 5 + 9 == 14
    assert result.iloc[1].added == 14

    # 5 * 9 == 45
    assert result.iloc[1].multiplied == 45

    save_model(model, model_path)

    # remove saved model
    rmtree(model_path.parent)


def test_load_and_predict(model_path):
    """
    Simple load of Model object using graph
    in test_model_save_and_load to predict
    and save out
    """
    # instantiate new model
    model = SampleModel()

    # save model
    save_model(model, model_path)

    del model

    # load test model
    model = load_model(model_path,
                       sample_model=True)

    # sample data
    new_data = [np.array([6, 0.3]),
                np.array([3, 0.1])]

    proba_threshold = 0.5

    # # "predict" (add, multiply), return binary since thresh given
    preds = model.predict(new_data,
                          proba_threshold)

    # 6 + 3 == 9 >= 0.5 --> True
    assert preds.iloc[0].added

    # 6 * 3 == 18 >= 0.5 --> True
    assert preds.iloc[0].multiplied

    # 0.3 + 0.1 == 0.4 <= 0.5 --> False
    assert not preds.iloc[1].added

    # 0.3 * 0.1 == 0.03 <= 0.5 --> False
    assert not preds.iloc[1].multiplied

    # delete test model
    rmtree(model_path.parent)
