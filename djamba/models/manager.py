from enum import Enum, EnumMeta
from pathlib import Path

from djamba.models.model import SampleModel
from djamba.models.winning_model import WinningModel


class GetItemMeta(EnumMeta):
    """Complicated override so that we can use hashing in ModelManager init.

    """
    def __getitem__(cls, name):
        for e in cls:
            if e.string == name:
                return e

        raise ValueError(f"Key '{name}' not in ModelName Enum.")


class ModelName(Enum, metaclass=GetItemMeta):
    """Allows easy control over which Model subclass to load, sample or winning.

        Args:
            string (str) : string for switching between test and default model
            model (Model) : Model subclass to instantiate into manager.

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
            model_path (str | Path) : path to model
                Required argument (for now)
            data_path (str | Path) : path to data
                Defaults to ``None``.
            pred_path (str | Path) : where predictions will be saved
                Defaults to ``None``.
            proba_threshold (float) : threshold for classification
                Defaults to ``None``.
            tempdir (str | Path) : path to temporary directory
                Defaults to ``None``.
            verbose (bool) : controls verbosity
                Defaults to ``True``.
            model_class (str) : controls whether sample class or production is used
                Defaults to "winning". Must be "winning" or "sample".
    """
    def __init__(self,
                 model_path,
                 data_path=None,
                 pred_path=None,
                 proba_threshold=None,
                 tempdir=None,
                 verbose=True,
                 model_class='winning'):

        self.model_path = Path(model_path)
        self.model_class = ModelName[model_class].model

        self.tempdir = tempdir
        self.model = self.model_class(model_path)
        self.proba_threshold = proba_threshold

        self.data_path = Path(data_path) if data_path else None
        self.pred_path = pred_path
        self.verbose = verbose

    def predict(self, data_path=None, pred_path=None):
        """
        Args:
            data_path (str | Path) : path to data, checks model_path attr if ``None``
            pred_path (str | Path) : where predictions will be saved

        Returns: DataFrame of predictions

        """

        if data_path is None:
            if self.data_path is not None:
                data_path = self.data_path
            else:
                raise FileNotFoundError("No data provided.")
        else:
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
