from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, validate_arguments
import yaml

from zamba.tests.sample_model import SampleModel
from zamba.models.cnnensemble_model import CnnEnsemble
from zamba.models.model import Model


class ModelClassEnum(str, Enum):
    cnnensemble = 'cnnensemble'
    sample = 'sample'
    custom = 'custom'


class ModelLibraryEnum(str, Enum):
    keras = 'keras'
    pytorch = 'pytorch'


# @validate_arguments
class TrainConfig(BaseModel):
    train_data: Path = Path("train_videos")
    val_data: Path = Path("val_videos")
    labels: Path = Path("labels.csv")
    model_path: Optional[Path] = None
    model_library: ModelLibraryEnum = 'keras'
    model_class: ModelClassEnum = 'custom'
    tempdir: Optional[Path] = None
    n_epochs: Optional[int] = 10
    height: Optional[int] = None
    width: Optional[int] = None
    augmentation: Optional[bool] = False
    early_stopping: Optional[bool] = False
    save_path: Optional[Path] = None


# @validate_arguments
class PredictConfig(BaseModel):
    data_path: Path = Path(".")
    model_path: Path = Path(".")
    model_class: ModelClassEnum = 'cnnensemble'
    pred_path: Optional[Path] = None
    output_class_names: Optional[bool] = False
    proba_threshold: Optional[float] = None
    tempdir: Optional[Path] = None
    verbose: Optional[bool] = False
    save: Optional[bool] = False
    model_kwargs: Optional[dict] = dict()
    predict_kwargs: Optional[dict] = dict()


# @validate_arguments
class FineTuneConfig(BaseModel):
    pass

# TODO: get validation to work
# @validate_arguments
class ModelConfig(BaseModel):
    train_config: Optional[TrainConfig] = None
    predict_config: Optional[PredictConfig] = None

    class Config:
        json_loads = yaml.safe_load


default_train = TrainConfig()
default_predict = PredictConfig()


class ModelManager(object):
    """Mediates loading, configuration, and logic of model calls.

        Args:
            model_path (str | Path) : path to model weights and architecture
                Required argument. Will be instantiated as Model object.
            proba_threshold (float) : probability threshold for classification
                Defaults to ``None``, in which case class probabilities are returned.
            tempdir (str | Path) : path to temporary directory
                If specific temporary directory is to be used, its path is passed here. Defaults to ``None``.
            model_class (str) : controls whether sample model class or production model class is used
                Defaults to "cnnensemble". Must be "cnnensemble", "sample", or "custom".
    """
    def __init__(self,
                 train_config=default_train,
                 predict_config=default_predict):

        self.train_config = train_config
        self.predict_config = predict_config
        self.logger = logging.getLogger(f"{__file__}")

    @staticmethod
    def from_config(config):
        if not isinstance(config, ModelManager):
            config = ModelConfig.parse_file(config)
        return ModelManager(**dict(config))


    def train(self):
        if self.train_config.model_class == 'custom':

            if not Path(self.train_config.model_path).exists():
                raise ValueError(f"{self.train_config.model_path} does not exist.")

            self.model = Model(
                model_path=self.train_config.model_path,
                model_library=self.train_config.model_library
            ).load()
        else:
            raise NotImplementedError('Currently only custom models can be trained.')

        self.model.fit(epochs=self.train_config.n_epochs)

    def predict(self):
        if self.predict_config.model_class == 'custom':
            self.model = Model(self.predict_config.model_path).load()

        else:
            model_dict = {
                'cnnensemble': CnnEnsemble,
                'sample': SampleModel
            }
            self.model = model_dict[self.predict_config.model_class.value](
                model_path=self.predict_config.model_path,
                tempdir=self.predict_config.tempdir,
                **self.predict_config.model_kwargs
            )

        data_path = self.predict_config.data_path
        pred_path = self.predict_config.pred_path

        data_paths = self.model.load_data(Path(data_path).expanduser().resolve())
        preds = self.model.predict(data_paths, **self.predict_config.predict_kwargs)

        # threshold if provided
        if self.predict_config.proba_threshold is not None:
            preds = preds >= self.predict_config.proba_threshold

        if self.predict_config.output_class_names:
            preds = preds.idxmax(axis=1)

        if self.predict_config.save:
            if pred_path is None:
                timestamp = datetime.now().isoformat()
                pred_path = Path('.', f'predictions-{Path(data_path).parts[-1]}-{timestamp}.csv')

            preds.to_csv(pred_path)

            if self.predict_config.verbose:
                self.logger.info(f"Wrote predictions to {pred_path}")

        if self.predict_config.verbose or self.predict_config.output_class_names:
            self.logger.info(preds)

        return preds

    def tune(self):
        """

        Returns:

        """
        pass
