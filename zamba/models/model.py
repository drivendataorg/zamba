import pickle
from pathlib import Path
from shutil import rmtree
import tempfile

import pandas as pd


try:
    from tensorflow.python import keras
except ImportError:
    msg = "Zamba must have tensorflow installed, run either `pip install zamba[cpu]` "\
          "or `pip install zamba[gpu]` " \
          "depending on what is available on your system."
    raise ImportError(msg)


class Model(object):
    """Abstract class implementing required methods for any model in the api.

        Args:
            model_path (str | Path) : path to model files
                Converted to ``Path`` object in ``__init__``. Defaults to ``None``.
            tempdir (str | Path) : path to temporary diretory
                Path to temporary directory, if used. Defaults to ``None``.

        Attributes:
            model_path (str | Path) : path to model files
                Converted to ``Path`` object in ``__init__``. Defaults to ``None``.
            tempdir (str | Path) : path to temporary diretory
                Path to temporary directory, if used. Defaults to ``None``.
            delete_tempdir (bool) : whether to clean up tempdir
                Clean up tempdir if used.

    """

    def __init__(self, model_path=None, tempdir=None, verbose=False):
        self.model_path = Path(model_path) if model_path is not None else None
        self.delete_tempdir = tempdir is None
        self.tempdir = Path(tempfile.mkdtemp()) if self.delete_tempdir else Path(tempdir)
        self.verbose = verbose

    def __del__(self):
        """ If we use the default temporary directory, clean this
            up when the model is removed.
        """
        if self.delete_tempdir:
            rmtree(self.tempdir)

    def predict(self, X):
        """

        Args:
            X: Input to model.

        Returns: DataFrame of class probabilities.

        """
        pass

    def fit(self, X, y):
        """Use the same architecture, but train the weights from scratch using
        the provided X and y.

        Args:
            X: training inputs
                Numpy arrays probably
            y: training labels
                Class labels

        Returns:

        """
        pass

    def finetune(self, X, y):
        """Finetune the network for a different task by keeping the
        trained weights, replacing the top layer with one that outputs
        the new classes, and re-training for a few epochs to have the
        model output the new classes instead.

        Args:
            X:
            y:

        Returns:

        """
        pass

    def save_model(self):
        """Save the model weights, checkpoints, to model_path.

        Returns:

        """


class SampleModel(Model):
    """Sample model for testing.

        Args:
            model_path:
            tempdir:
    """
    def __init__(self, model_path=None, tempdir=None, verbose=False):
        super().__init__(model_path, tempdir=tempdir)

        self.model = self._build_graph() if self.model_path is None else keras.models.load_model(self.model_path)
        self.verbose = verbose

    def _build_graph(self):
        """Simple keras graph for testing api.

        Takes two numbers, adds them, also multiplies them, outputs both results.

        Returns: keras model for testing

        """

        # build simple architecture to multiply two numbers
        w1 = keras.layers.Input(shape=(1,), name="w1")
        w2 = keras.layers.Input(shape=(1,), name="w2")

        add = keras.layers.add([w1, w2])
        mult = keras.layers.multiply([w1, w2])
        out = keras.layers.concatenate([add, mult])

        return keras.models.Model(inputs=[w1, w2], outputs=out)

    def predict(self, X):
        """

        Args:
            X (list | numpy array) : data for test computation

        Returns: DataFrame with two columns, ``added`` and ``multiplied``.

        """

        preds = self.model.predict(X)
        preds = pd.DataFrame(dict(added=preds[:, 0],
                                  multiplied=preds[:, 1]))
        return preds

    def save_model(self, path=None):
        """Save the SampleModel.

        If no path is passed, tries to use model_path attribute.

        Args:
            path:

        Returns:

        """

        # save to user-specified, or model's path
        path = Path(path) if path else None
        save_path = path or self.model_path
        if save_path is None:
            raise FileNotFoundError("Must provide save_path")

        # create if necessary
        save_path.parent.mkdir(exist_ok=True)

        # keras' save
        self.model.save(save_path, include_optimizer=False)

    def load_data(self, data_path):
        """SampleModel loads pickled data

        Args:
            data_path:

        Returns:

        """

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        return data
