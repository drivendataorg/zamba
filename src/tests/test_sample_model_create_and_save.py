import numpy as np

from src.src.models.io import save_model
from src.src.models.model import SampleModel

def test_create_and_save(model_path):

    # instantiate new model
    model = SampleModel(model_path)

    # build graph that multiplies 2 inputs
    model.build_graph()

    # create some sample data
    data = [np.array([4, 5]),
            np.array([8, 9])]

    # "predict" (add, multiply), return exact values since no thresh given
    result = model.predict_proba(data)

    # 4 + 8 == 12
    assert result.iloc[0].added == 12

    # 4 * 8 == 32
    assert result.iloc[0].multiplied == 32

    # 5 + 9 == 14
    assert result.iloc[1].added == 14

    # 5 * 9 == 45
    assert result.iloc[1].multiplied == 45

    save_model(model)
