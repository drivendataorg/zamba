import tempfile
import types

from pydantic import BaseModel
from pytorch_lightning.loggers import MLFlowLogger


def patch_mlflow_logger(logger: MLFlowLogger) -> MLFlowLogger:
    """Patch the mlflow logger to be able to write configs as yaml in artifact storage"""

    def log_config_artifact(self, config: BaseModel):
        # create a temporary file to store the config
        with tempfile.NamedTemporaryFile(delete=True, suffix=".yaml") as temp_file:
            with open(temp_file.name, "w") as f:
                f.write(config.json(indent=2, exclude={"labels"}))

            temp_file_path = temp_file.name

            # use log_artifact to save the config to the artifact storage
            self.experiment.log_artifact(self._run_id, temp_file_path, "zamba-config.yaml")

    logger.log_config_artifact = types.MethodType(log_config_artifact, logger)
    return logger
