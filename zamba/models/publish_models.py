import hashlib

from cloudpathlib import AnyPath, S3Path
from loguru import logger
import yaml

from zamba import MODELS_DIRECTORY
from zamba.models.config import WEIGHT_LOOKUP, ModelEnum


def publish_model(model_name, trained_model_dir):
    """
    Creates the files for the model folder in `official_models` and uploads the model to the three
    DrivenData public s3 buckets.

    Args:
        model_name (ModelEnum): Model name which will be folder name in `official_models`.
        trained_model_dir (AnyPath): Directory containing model checkpoint file,
            train_configuration.yaml, predict_configuration.yaml, config.yaml, hparams.yaml, and
            val_metrics.json.
    """
    checkpoints = list(AnyPath(trained_model_dir).rglob("*.ckpt"))
    if len(checkpoints) > 1:
        raise ValueError(
            f"{len(checkpoints)} were found in {trained_model_dir}. There can only be one. Checkpoints found include: {checkpoints}"
        )
    elif len(checkpoints) == 0:
        raise ValueError(f"No checkpoint files were found in {trained_model_dir}.")
    else:
        private_checkpoint = checkpoints[0]

    # configs are expected to be in the same folder as model checkpoint
    trained_model_dir = AnyPath(private_checkpoint).parent

    # copy over files from model directory
    logger.info(
        f"Copying over yaml and json files from {trained_model_dir} to {MODELS_DIRECTORY / model_name}."
    )
    for file in [
        "train_configuration.yaml",
        "predict_configuration.yaml",
        "config.yaml",
        "hparams.yaml",
        "val_metrics.json",
    ]:
        (AnyPath(trained_model_dir) / file).copy(MODELS_DIRECTORY / model_name)

    # prepare config for use in official models dir
    logger.info("Preparing official config file.")
    config_yaml = MODELS_DIRECTORY / model_name / "config.yaml"

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

            # e.g. european model is trained from a checkpoint; we want to expose final model
            # (model_name: european) not the base checkpoint
            if "checkpoint" in train_config.keys():
                train_config.pop("checkpoint")
                train_config["model_name"] = model_name

    official_config = dict(
        train_config=train_config,
        video_loader_config=config_dict["video_loader_config"],
        predict_config=dict(model_name=model_name),
    )

    # write out limited config
    logger.info(f"Writing out to {config_yaml}")
    with config_yaml.open("w") as f:
        yaml.dump(official_config, f, sort_keys=False)

    # hash config file to generate public filename for model
    hash_str = hashlib.sha1(str(official_config).encode("utf-8")).hexdigest()
    public_file_name = f"{model_name}_{hash_str}.ckpt"

    # upload to three public buckets
    for bucket in ["", "-eu", "-asia"]:
        public_checkpoint = S3Path(
            f"s3://drivendata-public-assets{bucket}/zamba_official_models/{public_file_name}"
        )
        logger.info(f"Uploading {private_checkpoint} to {public_checkpoint}")
        AnyPath(private_checkpoint).copy(public_checkpoint, force_overwrite_to_cloud=True)


if __name__ == "__main__":
    for model_name in ModelEnum.__members__.keys():
        private_checkpoint = WEIGHT_LOOKUP[model_name]
        logger.info(f"\n============\nPreparing {model_name} model\n============")
        publish_model(model_name, private_checkpoint)
