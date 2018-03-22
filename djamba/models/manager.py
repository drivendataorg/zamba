from enum import Enum, EnumMeta
from pathlib import Path

from djamba.models.model import SampleModel
from djamba.models.winning_model import WinningModel


class GetItemMeta(EnumMeta):
    def __getitem__(cls, name):
        for e in cls:
            if e.string == name:
                return e

        raise ValueError(f"Key '{name}' not in ModelName Enum.")


class ModelName(Enum, metaclass=GetItemMeta):
    WINNING = ('winning', WinningModel)
    SAMPLE = ('sample', SampleModel)

    def __init__(self, string, model):
        self.string = string
        self.model = model

    def __eq__(self, other):
        return other == self.value


class ModelManager(object):
    """
    Class to mediate loading,
    configuration, and logic
    of model calls.
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

        self.data_path = data_path
        self.pred_path = pred_path
        self.verbose = verbose

    def predict(self, data_path=None, pred_path=None):

        """
        Handle prediction
        """

        if data_path is None:
            if self.data_path is not None:
                data_path = self.data_path
            else:
                raise FileNotFoundError("No data provided.")

        data = self.model.load_data(data_path)

        preds = self.model.predict(data)

        if self.proba_threshold is None:
            return preds
        else:
            return preds >= self.proba_threshold

    def train(self):
        pass

    def tune(self):
        pass
