from datetime import datetime
from enum import Enum, EnumMeta
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pandas as pd

from zamba.models.cnnensemble.src import config
from zamba.models.model import SampleModel
from zamba.models.cnnensemble_model import CnnEnsemble


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

    WINNING = ('cnnensemble', CnnEnsemble)
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
                 model_class='cnnensemble'):

        self.model_path = Path(model_path)
        self.model_class = ModelName[model_class].model
        self._pass_sub_format = False if model_class == 'cnnensemble' else True

        self.tempdir = tempdir
        self.model = self.model_class(model_path)
        self.proba_threshold = proba_threshold

        self.verbose = verbose

    def predict(self, data_path, save=False, pred_path=None):
        """
        Args:
            data_path (str | Path) : path to input data
            pred_path (str | Path) : where predictions will be saved

        Returns: DataFrame of predictions

        """

        data_path = Path(data_path)

        # much of the cnn prediction code uses the submission format, requires correct video list
        self.make_submission_format_file(data_path)

        # cnn ensemble doesn't use simple data loader, samples do...
        if self.model_class != 'cnnensemble':
            data = self.model.load_data(data_path)
            preds = self.model.predict(data)
        else:
            preds = self.model.predict(data_path)

        # threshold if provided
        if self.proba_threshold is not None:
            preds = preds >= self.proba_threshold

        if save:
            if pred_path is None:
                timestamp = datetime.now().isoformat()
                pred_path = Path('.', f'predictions-{data_path.parts[-1]}-{timestamp}.csv')
            preds.to_csv(pred_path)

        return preds

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

    def make_submission_format_file(self, data_path):
        """This functions uses the files found in data_path to format a prediction DataFrame.
        The submission format csv saved here is used throught the prediction process for cnn ensemble.

        Args:
            data_path: path to data directory

        Returns:

        """

        # skip this if we're testing models other than cnn
        if self._pass_sub_format:
            return
        else:
            filelist = [f for f in listdir(data_path) if isfile(join(data_path, f))]
            classes = config.CLASSES

            current_sub_format = pd.DataFrame(index=pd.Index(filelist),
                                              columns=classes)
            current_sub_format.fillna(0.5, inplace=True)
            current_sub_format.index.name = 'filename'
            current_sub_format.to_csv(config.SUBMISSION_FORMAT)
