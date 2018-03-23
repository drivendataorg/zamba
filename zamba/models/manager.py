from enum import Enum, EnumMeta
from pathlib import Path

from zamba.models.model import SampleModel
from zamba.models.winning_model import WinningModel


class GetItemMeta(EnumMeta):
    """Complicated override so that we can use hashing in ModelManager init.

    """
    def __getitem__(cls, name):
        for e in cls:
            if e.string == name:
                return e

        raise ValueError(f"Key '{name}' not in ModelName Enum.")


class ModelName(Enum, metaclass=GetItemMeta):
    """Allows easy control over which Model subclass to load. To add a new model class, add a line like ``NEW_MODEL
    = ('new_model', NewModelClass)``

        Args:
            string (str) : string used to reference the model model
            model (Model) : model class to instantiate into manager

    """

    WINNING = ('winning', WinningModel)
    SAMPLE = ('sample', SampleModel)

    def __init__(self, string, model):
        self.string = string
        self.model = model

    def __eq__(self, other):
        return other == self.value


class ModelManager(object):
    """Mediates loading, configuration, and logic of model calls.

        Args:
            model_path (str | Path) : path to model weights and architecture
                Required argument. Will be instantiated as Model object.
            proba_threshold (float) : probability threshold for classification
                Defaults to ``None``, in which case class probabilities are returned.
            tempdir (str | Path) : path to temporary directory
                If specific temporary directory is to be used, its path is passed here. Defaults to ``None``.
            verbose (bool) : controls verbosity of prediction, training, and tuning methods
                Defaults to ``True`` in which case training, tuning or prediction progress will be logged.
            model_class (str) : controls whether sample model class or production model class is used
                Defaults to "winning". Must be "winning" or "sample".
    """
    def __init__(self,
                 model_path,
                 proba_threshold=None,
                 tempdir=None,
                 verbose=True,
                 model_class='winning'):

        self.model_path = Path(model_path)
        self.model_class = ModelName[model_class].model

        self.tempdir = tempdir
        self.model = self.model_class(model_path)
        self.proba_threshold = proba_threshold

        self.verbose = verbose

    def predict(self, data_path, pred_path=None):
        """
        Args:
            data_path (str | Path) : path to input data
            pred_path (str | Path) : where predictions will be saved

        Returns: DataFrame of predictions

        """

        data_path = Path(data_path)
        data = self.model.load_data(data_path)

        preds = self.model.predict(data)

        if self.proba_threshold is None:
            return preds
        else:
            return preds >= self.proba_threshold

    def train(self):
        """

        Returns:

        """
        pass

    def tune(self):
        """

        Returns:

        """
        pass
