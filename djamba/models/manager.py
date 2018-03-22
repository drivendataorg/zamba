from enum import Enum
from pathlib import Path

from djamba.models.model import SampleModel
from djamba.models.winning_model import WinningModel


class ModelName(Enum):
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

        for model in ModelName:
            if model_class.lower() == model.string:
                self.model_class = model.model

        self.tempdir = tempdir

        self.model = self.load_model(model_path) if self.model_path.exists() else None
        self.has_model = self.model is not None
        self.proba_thresh = proba_thresh

    def predict(self, X, data_path=None, output_path=None):

        """
        Handle prediction
        """

        preds = self.model.predict(X)

        if self.proba_thresh is None:
            return preds
        else:
            return preds >= self.proba_thresh

    def train(self):
        pass

    def tune(self):
        pass

    def add_model(self, model):
        if isinstance(model, self.model_class):
            self.model = model
            self.has_model = True
        else:
            raise TypeError(f"Model is type {type(model)}, ModelManager expecting type {self.model_class})")

    def load_model(self, model_path):
        """
        Return model object with saved keras graph
        """

        if not model_path.exists():
            raise FileNotFoundError

        return self.model_class(model_path)

    def save_model(self):
        """Only saves keras model currently"""

        if not self.has_model:
            raise AttributeError("Manager has no model.")

        # check for save paths
        if self.model.model_path is None:
            if self.model_path is not None:

                # create if necessary
                self.model_path.parent.mkdir(exist_ok=True)

                self.model.model_path = self.model_path
            else:
                raise AttributeError(f"model.model_path is {model.model_path}, please provide model_path")

        self.model.model.save(self.model.model_path)
