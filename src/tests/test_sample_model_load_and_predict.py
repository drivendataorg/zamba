import numpy as np

from src.src.models.io import load_model

def test_load_and_predict(model_path):
    """
    Simple load of Model object using graph
    in test_model_save_and_load to predict
    and save out
    """
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
    assert preds.iloc[0].added == True

    # 6 * 3 == 18 >= 0.5 --> True
    assert preds.iloc[0].multiplied == True

    # 0.3 + 0.1 == 0.4 <= 0.5 --> False
    assert preds.iloc[1].added == False

    # 0.3 * 0.1 == 0.03 <= 0.5 --> False
    assert preds.iloc[1].multiplied == False
