from pathlib import Path

from djamba.models.model import Model, SampleModel


def load_model(modelpath, sample_model=False):
    """
    Return model object with saved keras graph
    """

    modelpath = Path(modelpath)
    if not modelpath.exists():
        raise FileNotFoundError

    if sample_model:
        return SampleModel(modelpath)
    else:
        return Model(modelpath)


def save_model(model, model_path=None):
    """Only saves keras model currently"""

    # check for save paths
    if model.model_path is None:
        if model_path is not None:

            # create if necessary
            model_path.parent.mkdir(exist_ok=True)

            model.model_path = model_path
        else:
            raise AttributeError(f"model.model_path is {model.model_path}, please provide model_path")

    model.model.save(model.model_path)
