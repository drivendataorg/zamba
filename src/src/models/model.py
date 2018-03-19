from pathlib import Path
from shutil import rmtree
import tempfile

import pandas as pd
from tensorflow.python import keras


class Model(object):
    def __init__(self, modeldir=None, tempdir=None):
        self.modeldir = Path(modeldir) if modeldir is not None else None
        self.delete_tempdir = tempdir is None
        self.tempdir = Path(tempfile.mkdtemp()) if self.delete_tempdir else Path(tempdir)

    def __del__(self):
        """ If we use the default temporary directory, clean this
            up when the model is removed.
        """
        if self.delete_tempdir:
            rmtree(self.tempdir)

    def predict(self, X):
        """
        Predict class probabilities
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


class SampleModel(Model):
    def __init__(self, modeldir=None, tempdir=None):
        super().__init__(modeldir, tempdir=tempdir)

        self.model = self._build_graph() if self.modeldir is None else keras.models.load_model(self.modeldir)

    def _build_graph(self):

        # build simple architecture to multiply two numbers
        w1 = keras.layers.Input(shape=(1,), name="w1")
        w2 = keras.layers.Input(shape=(1,), name="w2")

        add = keras.layers.add([w1, w2])
        mult = keras.layers.multiply([w1, w2])
        out = keras.layers.concatenate([add, mult])

        return keras.models.Model(inputs=[w1, w2], outputs=out)

    def predict(self, X, proba_threshold=None):
        """
        Predict class probabilities
        """

        predictions = self.model.predict(X)
        preds_df = pd.DataFrame(dict(added=predictions[:, 0],
                                     multiplied=predictions[:, 1]))

        if proba_threshold is None:
            return preds_df
        else:
            return preds_df >= proba_threshold
