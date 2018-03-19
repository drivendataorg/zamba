from pathlib import Path

from src.src.models.model import Model, SampleModel


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
    if model.modeldir is None:
        if model_path is not None:

            # create if necessary
            model_path.parent.mkdir(exist_ok=True)

            model.modeldir = model_path
        else:
            raise AttributeError(f"model.modeldir is {model.modeldir}, please provide model_path")

    model.model.save(model.modeldir)
