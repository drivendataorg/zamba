import pickle
from pathlib import Path

from tensorflow import keras
import pandas as pd

from zamba.models.model import Model


class SampleModel(Model):
    """Sample model for testing.

        Args:
            model_path:
            tempdir:
    """
    def __init__(self, tempdir=None, model_path=None):
        super().__init__(tempdir=tempdir, model_path=model_path)
        self.model = self._build_graph() if self.model_path is None else keras.models.load_model(Path(self.model_path))

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

    @classmethod
    def load(cls, model_path):
        return cls(model_path=model_path)

    def save_model(self, path=None):
        """Save the SampleModel.

        If no path is passed, tries to use model_path attribute.

        Args:
            path:

        Returns:

        """

        # save to user-specified, or model's path
        if path is not None:
            save_path = Path(path)
        elif self.model_path is not None:
            save_path = Path(self.model_path)
        else:
            raise FileNotFoundError("Must provide save_path")

        # create if necessary
        save_path.parent.mkdir(exist_ok=True)

        # keras' save
        self.model.save(save_path, include_optimizer=False)

    def load_data(self, data_path):
        """SampleModel loads pickled data

        Args:
            data_path (pathlib.Path)

        Returns:

        """
        with data_path.open("rb") as f:
            data = pickle.load(f)

        return data
