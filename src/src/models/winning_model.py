from pathlib import Path
from shutil import rmtree
import tempfile

import pandas as pd
import tensorflow as tf

from .model import Model


class WinningModel(Model):
    def __init__(self, modeldir, tempdir=None):
        # use the model object's defaults
        super().__init__(modeldir, tempdir=tempdir)

    def predict(self, X):
        """ Predict class probabilities for each input, X
        """
        pass

    def fit(self, X, y):
        """ Use the same architecture, but train the weights from scratch using
            the provided X and y.
        """
        pass

    def finetune(self, X, y):
        """ Finetune the network for a different task by keeping the
            trained weights, replacing the top layer with one that outputs
            the new classes, and re-training for a few epochs to have the
            model output the new classes instead.
        """
        pass
