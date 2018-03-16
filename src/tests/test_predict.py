from pathlib import Path
from src.src.models.io import load_model

def test_predict_modelpath(model_subdir, predsout):
    """
    Simple load of Model object using graph
    in test_model_save_and_load to predict
    and save out
    """
    # load test model
    model = load_model(model_subdir)

    # sample preds
    preds = model.predict_proba(tmp=None) >= 0.5

    # save
    preds.to_csv(
        Path(str(predsout)).resolve(),
        index_label='id',
    )
