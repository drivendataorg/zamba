from pathlib import Path
from shutil import rmtree
import tempfile


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
    def __init__(self, model_path=None, model_library=None, tempdir=None):

        self.model_path = model_path
        self.model_library = model_library
        self.delete_tempdir = tempdir is None
        self.tempdir = Path(tempfile.mkdtemp(prefix="zamba_")) if self.delete_tempdir else Path(tempdir)

    def __del__(self):
        """ If we use the default temporary directory, clean this
            up when the model is removed.
        """
        if self.delete_tempdir:
            rmtree(self.tempdir)

    def load(self):
        if self.model_path is not None:
            if self.model_library == 'keras':
                import keras
                return keras.models.load_model(str(self.model_path))
            elif self.model_library == 'pytorch':
                import torch
                return torch.load(str(self.model_path))
            else:
                raise NotImplementedError("Currently, only Keras or PyTorch models can be loaded.")

    def load_data(self, data_path):
        input_paths = [
            path for path in data_path.glob('**/*')
            if not path.is_dir() and not path.name.startswith(".")
        ]

        return input_paths

    def predict(self, X):
        """

        Args:
            X: Input to model.

        Returns: DataFrame of class probabilities.

        """
        pass

    def fit(self):
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
        pass
