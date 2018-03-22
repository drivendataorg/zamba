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
    def __init__(self, model_path, model_class='winning', tempdir=None, proba_thresh=None):

        self.model_path = Path(model_path)
        self.model_class = ModelName[model_class].model

        self.tempdir = tempdir
        self.model = self.model_class(model_path)
        self.proba_thresh = proba_thresh

    def predict(self, data_path=None, output_path=None):

        """
        Handle prediction
        """

        data = self.model.load_data(data_path)

        preds = self.model.predict(data)

        if self.proba_thresh is None:
            return preds
        else:
            return preds >= self.proba_thresh

    def train(self):
        pass

    def tune(self):
        pass
