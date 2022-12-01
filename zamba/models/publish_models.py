import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request

from cloudpathlib import AnyPath, S3Path
from loguru import logger
import yaml

from zamba import MODELS_DIRECTORY
from zamba.models.config import WEIGHT_LOOKUP, ModelEnum
from zamba.models.densepose import MODELS as DENSEPOSE_MODELS
from zamba.models.depth_estimation import MODELS as DEPTH_MODELS


def get_model_only_params(full_configuration, subset="train_config"):
    """Return only params that are not data or machine specific.
    Used for generating official configs.
    """
    if subset == "train_config":
        config = full_configuration[subset]
        for key in [
            "data_dir",
            "dry_run",
            "batch_size",
            "auto_lr_find",
            "gpus",
            "num_workers",
            "max_epochs",
            "weight_download_region",
            "split_proportions",
            "save_dir",
            "overwrite",
            "skip_load_validation",
            "from_scratch",
            "model_cache_dir",
            "use_default_model_labels",
            "predict_all_zamba_species",
        ]:
            try:
                config.pop(key)
            except:  # noqa: E722
                continue

    elif subset == "video_loader_config":
        config = full_configuration[subset]

        if "megadetector_lite_config" in config.keys():
            config["megadetector_lite_config"].pop("device")

        for key in ["cache_dir", "cleanup_cache"]:
            config.pop(key)

    return config


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

    # make model directory
    (MODELS_DIRECTORY / model_name).mkdir(exist_ok=True, parents=True)

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

    # start with full train configuration
    with (MODELS_DIRECTORY / model_name / "train_configuration.yaml").open() as f:
        train_configuration_full_dict = yaml.safe_load(f)

    # get limited train config
    train_config = get_model_only_params(train_configuration_full_dict, subset="train_config")

    # e.g. european model is trained from a checkpoint; we want to expose final model
    # (model_name: european) not the base checkpoint
    if "checkpoint" in train_config.keys():
        train_config.pop("checkpoint")
        train_config["model_name"] = model_name

    official_config = dict(
        train_config=train_config,
        video_loader_config=get_model_only_params(
            train_configuration_full_dict, subset="video_loader_config"
        ),
        predict_config=dict(model_name=model_name),
    )

    # hash train_configuration to generate public filename for model
    hash_str = hashlib.sha1(str(train_configuration_full_dict).encode("utf-8")).hexdigest()[:10]
    public_file_name = f"{model_name}_{hash_str}.ckpt"

    # add that to official config
    official_config["public_checkpoint"] = public_file_name

    # write out official config
    config_yaml = MODELS_DIRECTORY / model_name / "config.yaml"
    logger.info(f"Writing out to {config_yaml}")
    with config_yaml.open("w") as f:
        yaml.dump(official_config, f, sort_keys=False)

    upload_to_all_public_buckets(private_checkpoint, public_file_name)


def upload_to_all_public_buckets(file, public_file_name):
    # upload to three public buckets
    for bucket in ["", "-eu", "-asia"]:
        public_checkpoint = S3Path(
            f"s3://drivendata-public-assets{bucket}/zamba_official_models/{public_file_name}"
        )
        if public_checkpoint.exists():
            logger.info(f"Skipping since {public_checkpoint} exists.")
        else:
            logger.info(f"Uploading {file} to {public_checkpoint}")
            public_checkpoint.upload_from(file, force_overwrite_to_cloud=True)


if __name__ == "__main__":
    for model_name in ModelEnum.__members__.keys():
        private_checkpoint = WEIGHT_LOOKUP[model_name]
        logger.info(f"\n============\nPreparing {model_name} model\n============")
        publish_model(model_name, private_checkpoint)

    for name, model in DEPTH_MODELS.items():
        logger.info(f"\n============\nPreparing {name} model\n============")
        # upload to the zamba buckets, renaming to model["weights"]
        upload_to_all_public_buckets(S3Path(model["private_weights_url"]), model["weights"])

    for name, model in DENSEPOSE_MODELS.items():
        logger.info(f"\n============\nPreparing DensePose model: {name}\n============")

        if S3Path(
            f"s3://drivendata-public-assets/zamba_official_models/{model['weights']}"
        ).exists():
            logger.info("Skipping since model exists on main public S3 bucket.")
            continue

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tmp_download_path = tmpdir / model["weights"]

            # download from the facebook servers
            logger.info(f"Downloading weights: {model['densepose_weights_url']}")
            urllib.request.urlretrieve(model["densepose_weights_url"], tmp_download_path)

            # upload to the zamba buckets, renaming to model["weights"]
            logger.info(f"Uploading to zamba buckets: {model['weights']}")
            upload_to_all_public_buckets(tmp_download_path, model["weights"])

            # remove local temp file that was downloaded
            tmp_download_path.unlink()
