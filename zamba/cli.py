from pathlib import Path

from loguru import logger
from pydantic.error_wrappers import ValidationError
import typer
import yaml

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.config import (
    MODEL_MAPPING,
    ModelConfig,
    ModelEnum,
    PredictConfig,
    TrainConfig,
)
from zamba_algorithms.models.model_manager import ModelManager
from zamba_algorithms.models.utils import RegionEnum


app = typer.Typer()


@app.command()
def train(
    data_dir: Path = typer.Option(None, exists=True, help="Path to folder containing videos."),
    labels: Path = typer.Option(None, exists=True, help="Path to csv containing video labels."),
    model: ModelEnum = typer.Option(
        "time_distributed",
        help="Model to train. Model will be superseded by checkpoint if provided.",
    ),
    checkpoint: Path = typer.Option(
        None,
        exists=True,
        help="Model checkpoint path to use for training. If provided, model is not required.",
    ),
    config: Path = typer.Option(
        None,
        exists=True,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    batch_size: int = typer.Option(None, help="Batch size to use for training."),
    gpus: int = typer.Option(
        None,
        help="Number of GPUs to use for training. If not specifiied, will use all GPUs found on machine.",
    ),
    dry_run: bool = typer.Option(
        None,
        help="Runs one batch of train and validation to check for bugs.",
    ),
    save_dir: Path = typer.Option(
        None,
        help="Directory in which to save model checkpoint and configuration file. If not specified, will save to a folder called 'zamba_{model_name}' in your working directory.",
    ),
    num_workers: int = typer.Option(
        None,
        help="Number of subprocesses to use for data loading.",
    ),
    weight_download_region: RegionEnum = typer.Option(
        None, help="Server region for downloading weights."
    ),
    cache_dir: Path = typer.Option(
        None,
        exists=False,
        help="Path to directory for model weights. Alternatively, specify with environment variable `ZAMBA_CACHE_DIR`. If not specified, user's cache directory is used.",
    ),
    skip_load_validation: bool = typer.Option(
        None,
        help="Skip check that verifies all videos can be loaded prior to training. Only use if you're very confident all your videos can be loaded.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation of configuration and proceed right to training.",
    ),
):
    """Train a model on your labeled data.

    If an argument is specified in both the command line and in a yaml file, the command line input will take precedence.
    """
    if config is not None:
        with config.open() as f:
            config_dict = yaml.safe_load(f)
        config_file = config
    else:
        with MODEL_MAPPING[model.value]["config"].open() as f:
            config_dict = yaml.safe_load(f)
        config_file = None

    if "video_loader_config" in config_dict.keys():
        video_loader_dict = config_dict["video_loader_config"]
    else:
        video_loader_dict = dict()

    train_dict = config_dict["train_config"]

    # override if any command line arguments are passed
    if data_dir is not None:
        train_dict["data_directory"] = data_dir

    if labels is not None:
        train_dict["labels"] = labels

    if model != "time_distributed":
        train_dict["model_name"] = model

    if checkpoint is not None:
        train_dict["checkpoint"] = checkpoint

    if batch_size is not None:
        train_dict["batch_size"] = batch_size

    if gpus is not None:
        train_dict["gpus"] = gpus

    if dry_run is not None:
        train_dict["dry_run"] = dry_run

    if save_dir is not None:
        train_dict["save_directory"] = save_dir

    if num_workers is not None:
        train_dict["num_workers"] = num_workers

    if weight_download_region is not None:
        train_dict["weight_download_region"] = weight_download_region

    if cache_dir is not None:
        train_dict["cache_dir"] = cache_dir

    if skip_load_validation is not None:
        train_dict["skip_load_validation"] = skip_load_validation

    try:
        manager = ModelManager(
            ModelConfig(
                video_loader_config=VideoLoaderConfig(**video_loader_dict),
                train_config=TrainConfig(**train_dict),
            )
        )
    except ValidationError as ex:
        logger.error("Invalid configuration.")
        raise typer.Exit(ex)

    config = manager.config

    # get species to confirm
    spacer = "\n\t- "
    species = spacer + spacer.join(
        sorted(
            [
                c.split("species_", 1)[1]
                for c in config.train_config.labels.filter(regex="species").columns
            ]
        )
    )

    msg = f"""The following configuration will be used for training:

    Config file: {config_file}
    Data directory: {data_dir if data_dir is not None else config_dict["train_config"].get("data_directory")}
    Labels csv: {labels if labels is not None else config_dict["train_config"].get("labels")}
    Species: {species}
    Model name: {config.train_config.model_name}
    Checkpoint: {checkpoint if checkpoint is not None else config_dict["train_config"].get("checkpoint")}
    Weight download region: {config.train_config.weight_download_region}
    Batch size: {config.train_config.batch_size}
    Number of workers: {config.train_config.num_workers}
    GPUs: {config.train_config.gpus}
    Dry run: {config.train_config.dry_run}
    Save directory: {config.train_config.save_directory}
    Cache directory: {config.train_config.cache_dir}
    """

    if yes:
        typer.echo(f"{msg}\n\nSkipping confirmation and proceeding to train.")
    else:
        yes = typer.confirm(
            f"{msg}\n\nIs this correct?",
            abort=False,
            default=True,
        )

    if yes:
        # kick off training
        manager.train()


@app.command()
def predict(
    data_dir: Path = typer.Option(None, exists=True, help="Path to folder containing videos."),
    filepaths: Path = typer.Option(
        None, exists=True, help="Path to csv containing `filepath` column with videos."
    ),
    model: ModelEnum = typer.Option(
        "time_distributed",
        help="Model to use for inference. Model will be superseded by checkpoint if provided.",
    ),
    checkpoint: Path = typer.Option(
        None,
        exists=True,
        help="Model checkpoint path to use for inference. If provided, model is not required.",
    ),
    gpus: int = typer.Option(
        None,
        help="Number of GPUs to use for inference. If not specifiied, will use all GPUs found on machine.",
    ),
    batch_size: int = typer.Option(None, help="Batch size to use for training."),
    save: bool = typer.Option(
        None,
        help="Whether to save out predictions to a csv file. If you want to specify the location of the csv, use save_path instead.",
    ),
    save_path: Path = typer.Option(
        None,
        help="Full path for prediction CSV file. Any needed parent directories will be created.",
    ),
    dry_run: bool = typer.Option(None, help="Runs one batch of inference to check for bugs."),
    config: Path = typer.Option(
        None,
        exists=True,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    proba_threshold: float = typer.Option(
        None,
        help="Probability threshold for classification between 0 and 1. If specified binary predictions "
        "are returned with 1 being greater than the threshold, 0 being less than or equal to. If not "
        "specified, probabilities between 0 and 1 are returned.",
    ),
    output_class_names: bool = typer.Option(
        None,
        help="If True, we just return a video and the name of the most likely class. If False, "
        "we return a probability or indicator (depending on --proba_threshold) for every "
        "possible class.",
    ),
    num_workers: int = typer.Option(
        None,
        help="Number of subprocesses to use for data loading.",
    ),
    weight_download_region: RegionEnum = typer.Option(
        None, help="Server region for downloading weights."
    ),
    cache_dir: Path = typer.Option(
        None,
        exists=False,
        help="Path to directory for model weights. Alternatively, specify with environment variable `ZAMBA_CACHE_DIR`. If not specified, user's cache directory is used.",
    ),
    skip_load_validation: bool = typer.Option(
        None,
        help="Skip check that verifies all videos can be loaded prior to inference. Only use if you're very confident all your videos can be loaded.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation of configuration and proceed right to prediction.",
    ),
):
    """Identify species in a video.

    This is a command line interface for prediction on camera trap footage. Given a path to camera
    trap footage, the predict function use a deep learning model to predict the presence or absense of
    a variety of species of common interest to wildlife researchers working with camera trap data.

    If an argument is specified in both the command line and in a yaml file, the command line input will take precedence.
    """
    if config is not None:
        with config.open() as f:
            config_dict = yaml.safe_load(f)
        config_file = config
    else:
        with MODEL_MAPPING[model.value]["config"].open() as f:
            config_dict = yaml.safe_load(f)
        config_file = None

    if "video_loader_config" in config_dict.keys():
        video_loader_dict = config_dict["video_loader_config"]
    else:
        video_loader_dict = dict()

    predict_dict = config_dict["predict_config"]

    # override if any command line arguments are passed
    if data_dir is not None:
        predict_dict["data_directory"] = data_dir

    if filepaths is not None:
        predict_dict["filepaths"] = filepaths

    if model != "time_distributed":
        predict_dict["model_name"] = model

    if checkpoint is not None:
        predict_dict["checkpoint"] = checkpoint

    if batch_size is not None:
        predict_dict["batch_size"] = batch_size

    if gpus is not None:
        predict_dict["gpus"] = gpus

    if dry_run is not None:
        predict_dict["dry_run"] = dry_run

    if save is not None:
        predict_dict["save"] = save

    # save path takes precedence over save
    if save_path is not None:
        predict_dict["save"] = save_path

    if proba_threshold is not None:
        predict_dict["proba_threshold"] = proba_threshold

    if output_class_names is not None:
        predict_dict["output_class_names"] = output_class_names

    if num_workers is not None:
        predict_dict["num_workers"] = num_workers

    if weight_download_region is not None:
        predict_dict["weight_download_region"] = weight_download_region

    if cache_dir is not None:
        predict_dict["cache_dir"] = cache_dir

    if skip_load_validation is not None:
        predict_dict["skip_load_validation"] = skip_load_validation

    try:
        manager = ModelManager(
            ModelConfig(
                video_loader_config=VideoLoaderConfig(**video_loader_dict),
                predict_config=PredictConfig(**predict_dict),
            )
        )
    except ValidationError as ex:
        logger.error("Invalid configuration.")
        raise typer.Exit(ex)

    config = manager.config

    msg = f"""The following configuration will be used for inference:

    Config file: {config_file}
    Data directory: {data_dir if data_dir is not None else config_dict["predict_config"].get("data_directory")}
    Filepath csv: {filepaths if filepaths is not None else config_dict["predict_config"].get("filepaths")}
    Model: {config.predict_config.model_name}
    Checkpoint: {checkpoint if checkpoint is not None else config_dict["predict_config"].get("checkpoint")}
    Batch size: {config.predict_config.batch_size}
    Number of workers: {config.predict_config.num_workers}
    GPUs: {config.predict_config.gpus}
    Dry run: {config.predict_config.dry_run}
    Save: {config.predict_config.save}
    Proba threshold: {config.predict_config.proba_threshold}
    Output class names: {config.predict_config.output_class_names}
    Weight download region: {config.predict_config.weight_download_region}
    Cache directory: {config.predict_config.cache_dir}
    """

    if yes:
        typer.echo(f"{msg}\n\nSkipping confirmation and proceeding to prediction.")
    else:
        yes = typer.confirm(
            f"{msg}\n\nIs this correct?",
            abort=False,
            default=True,
        )

    if yes:
        # kick off prediction
        manager.predict()


if __name__ == "__main__":

    app()
