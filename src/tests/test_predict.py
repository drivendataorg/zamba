from pathlib import Path
from src.src.models.io import load_model

def test_predict_modelpath(model_subdir, predsout):
    """
    Simple load of Model object using graph
    in test_model_save_and_load to predict
    and save out
    """
    # load test model
    model = load_model(model_subdir,
                       sample_model=True)

    # sample data: {w1: 0, w2: 1, bias: 2}
    new_feed_dict = model.make_sample_data()

    # sample calc using op: (w1 + w2) * bias
    probs = model.predict_proba(new_feed_dict)

    # (w1 + w2) * bias : (0 + 1) * 2
    preds = probs >= 0.5

    # 2 >= 0.5 == True
    assert preds.iloc[0]['output'] == True
