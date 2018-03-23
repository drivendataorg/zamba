import pickle
from pathlib import Path
from shutil import rmtree
import tempfile

import pandas as pd
from tensorflow.python import keras


class Model(object):
    def __init__(self, model_path=None, tempdir=None):
        self.model_path = Path(model_path) if model_path is not None else None
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
        Predict class probabilities.
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

    def save_model(self):
        """Save the model weights, checkpoints, to model_path.
        """


class SampleModel(Model):
    def __init__(self, model_path=None, tempdir=None):
        super().__init__(model_path, tempdir=tempdir)

        self.model = self._build_graph() if self.model_path is None else keras.models.load_model(self.model_path)

    def _build_graph(self):

        # build simple architecture to multiply two numbers
        w1 = keras.layers.Input(shape=(1,), name="w1")
        w2 = keras.layers.Input(shape=(1,), name="w2")

        add = keras.layers.add([w1, w2])
        mult = keras.layers.multiply([w1, w2])
        out = keras.layers.concatenate([add, mult])

        return keras.models.Model(inputs=[w1, w2], outputs=out)

    def predict(self, X):
        """
        Predict class probabilities
        """

        preds = self.model.predict(X)
        preds = pd.DataFrame(dict(added=preds[:, 0],
                                  multiplied=preds[:, 1]))
        return preds

    def save_model(self, path=None):
        """Only saves keras model currently"""

        # save to user-specified, or model's path
        path = Path(path) if path else None
        save_path = path or self.model_path
        if save_path is None:
            raise FileNotFoundError("Must provide save_path")

        # create if necessary
        save_path.parent.mkdir(exist_ok=True)

        # keras' save
        self.model.save(save_path)

    def load_data(self, data_path):
        """SampleModel loads pickled data"""

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        return data
