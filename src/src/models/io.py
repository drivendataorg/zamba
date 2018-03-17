from pathlib import Path

import tensorflow as tf

from src.src.models.model import Model, SampleModel

def load_model(modeldir, sample_model=False):
    """
    Return model object with saved tensorflow graph
    """

    modeldir = Path(modeldir)
    if not modeldir.exists():
        raise FileNotFoundError

    suffixes = [str(f.suffix) for f in modeldir.resolve().iterdir()]
    if ".meta" not in suffixes:
        raise FileNotFoundError("Model requires metagraph")

    stems = [str(f.stem) for f in modeldir.resolve().iterdir()]
    if "checkpoint" not in stems:
        msg = "Expected model weights to be in checkpoint dir"
        raise FileNotFoundError(msg)

    if sample_model:
        return SampleModel(modeldir)
    else:
        return Model(modeldir)
