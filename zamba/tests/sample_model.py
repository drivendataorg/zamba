import pickle
from pathlib import Path

from tensorflow import keras
import pandas as pd

from zamba.models.keras_model import KerasModel


class SampleModel(KerasModel):
    """Sample model for testing.
    """
    def __init__(self, tempdir=None, model_load_path=None, model_save_path=None):
        super().__init__(tempdir=tempdir, model_save_path=model_save_path)
        self.model = self._build_graph() if model_load_path is None else keras.models.load_model(Path(model_load_path))

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

    def load_data(self, data_path):
        """SampleModel loads pickled data

        Args:
            data_path (pathlib.Path)

        Returns:

        """
        with data_path.open("rb") as f:
            data = pickle.load(f)

        return data

    @classmethod
    def from_disk(cls, path):
        return cls(model_load_path=path)
