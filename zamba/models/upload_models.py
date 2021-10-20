import hashlib
from pathlib import Path

from cloudpathlib import AnyPath, S3Path
from loguru import logger
import yaml

from zamba import MODELS_DIRECTORY
from zamba.models.config import WEIGHT_LOOKUP


def publish_model(published_model_name, private_checkpoint):
    # configs are expected to be in the same folder as model checkpoint
    trained_model_dir = AnyPath(private_checkpoint).parent

    # copy over files from model directory
    for file in [
        "train_configuration.yaml",
        "predict_configuration.yaml",
        "config.yaml",
        "hparams.yaml",
        "val_metrics.json",
    ]:

        (AnyPath(trained_model_dir) / file).copy(MODELS_DIRECTORY / published_model_name)

    # prepare config for use in official models dir
    config_yaml = (MODELS_DIRECTORY / published_model_name / "config.yaml")

    with config_yaml.open() as f:
        config_dict = yaml.safe_load(f)

    train_config = dict()
    for key in config_dict["train_config"].keys():
        # only keep config values that are not data or machine specific
        if key in [
            "model_name",
            "backbone_finetune_config",
            "early_stopping_config",
            "scheduler_config",
            "predict_all_zamba_species",
            "checkpoint",
        ]:
            train_config[key] = config_dict["train_config"][key]

    official_config = dict(
        train_config=train_config,
        video_loader_config=config_dict["video_loader_config"],
        predict_config=dict(model_name=published_model_name),
    )

    # write out limited config
    with config_yaml.open("w") as f:
        yaml.dump(official_config, f, sort_keys=False)

    # hash config file to generate public filename for model
    hash_str = hashlib.sha1(str(config_dict).encode("utf-8")).hexdigest()
    public_file_name = f"{published_model_name}_{hash_str}.ckpt"

    # upload to three public buckets
    for bucket in ["", "-eu", "-asia"]:
        public_checkpoint = S3Path(f"s3://drivendata-public-assets{bucket}/{public_file_name}")
        logger.info(f"Uploading {private_checkpoint} to {public_checkpoint}")
        private_checkpoint.copy(public_checkpoint, force_overwrite_to_cloud=True)


if __name__ == "__main__":
    for model_name, private_model_dir in WEIGHT_LOOKUP.items():
        if model_name == "slowfast":
            publish_model(model_name, private_model_dir)
