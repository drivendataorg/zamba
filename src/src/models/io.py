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


def save_model(model):
    """Only saves keras model currently"""

    # save keras model
    model.model.save(model.modeldir)
