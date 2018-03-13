from pathlib import Path

import tensorflow as tf

from src.src.models.model import Model

def load_model(modeldir):
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

    return Model(modeldir)
