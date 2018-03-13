from pathlib import Path

import tensorflow as tf

from src.src.models.model import Model

def load_model(modeldir):
    """
    Return model object with saved tensorflow graph
    """

    # Check that modelpath dir exists
    modeldir = Path(modeldir)
    assert modeldir.exists()

    # Check that modelpath dir contains metagraph (to load network)
    suffixes = [str(f.suffix) for f in modeldir.resolve().iterdir()]
    assert ".meta" in suffixes

    # Check that modelpath dir contains checkpoint (to load weights)
    stems = [str(f.stem) for f in modeldir.resolve().iterdir()]
    assert "checkpoint" in stems

    return Model(modeldir)
