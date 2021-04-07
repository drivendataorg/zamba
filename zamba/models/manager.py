from datetime import datetime
import logging
from pathlib import Path

from zamba.tests.sample_model import SampleModel
from zamba.models.cnnensemble_model import CnnEnsemble
from zamba.models.model import Model
from zamba.models.config import TrainConfig, PredictConfig


default_train = TrainConfig()
default_predict = PredictConfig()


class ModelManager(object):
    """Mediates loading, configuration, and logic of model calls.

        Args:
            train_config (TrainConfig) : Configuration for model training.
            predict_config (PredictConfig): Configuration for inference.
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
