from enum import Enum, EnumMeta
from pathlib import Path
from shutil import rmtree
import tempfile
from typing import Optional

from pydantic import BaseModel
import yaml


# class ModelEnum(str, Enum):
#     CnnEnsemble = 'cnnensemble'
#     SampleModel = 'sample'


class ModelConfig(BaseModel):
    model_path: Path = Path(".")
    tempdir: Optional[Path]
    verbose: bool = False
     # proba_threshold: float = None
    # output_class_names: bool = False
    # model_class: ModelEnum = 'cnnensemble'
    # model_kwargs: dict = dict()

    class Config:
        json_loads = yaml.safe_load


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
    def __init__(self, model_path=Path('.'), tempdir=None, verbose=False):

        self.model_path = model_path
        self.delete_tempdir = tempdir is None
        self.tempdir = Path(tempfile.mkdtemp(prefix="zamba_")) if self.delete_tempdir else Path(tempdir)
        self.verbose = verbose

    def __del__(self):
        """ If we use the default temporary directory, clean this
            up when the model is removed.
        """
        if self.delete_tempdir:
            rmtree(self.tempdir)

    @staticmethod
    def from_config(config):
        if not isinstance(config, ModelConfig):
            config = ModelConfig.parse_file(config)
        return Model(**config.dict())

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
