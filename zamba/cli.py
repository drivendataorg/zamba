from pathlib import Path
from typing import Optional
import warnings

from loguru import logger
from pydantic.error_wrappers import ValidationError
import typer
import yaml

from zamba.data.video import VideoLoaderConfig
from zamba.models.config import (
    ModelConfig,
    ModelEnum,
    PredictConfig,
    TrainConfig,
)
from zamba import MODELS_DIRECTORY
from zamba.models.densepose import DensePoseConfig, DensePoseOutputEnum
from zamba.models.depth_estimation import DepthEstimationConfig
from zamba.models.model_manager import ModelManager
from zamba.models.utils import RegionEnum
from zamba.version import __version__


app = typer.Typer(pretty_exceptions_show_locals=False)


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
        help="An optional directory in which to save the model checkpoint and configuration file. If not specified, will save to a `version_n` folder in your working directory.",
    ),
    num_workers: int = typer.Option(
        None,
        help="Number of subprocesses to use for data loading.",
    ),
    weight_download_region: RegionEnum = typer.Option(
        None, help="Server region for downloading weights."
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
        with (MODELS_DIRECTORY / f"{model.value}/config.yaml").open() as f:
            config_dict = yaml.safe_load(f)
        config_file = None

    if "video_loader_config" in config_dict.keys():
        video_loader_config = VideoLoaderConfig(**config_dict["video_loader_config"])
    else:
        video_loader_config = None

    train_dict = config_dict["train_config"]

    # override if any command line arguments are passed
    if data_dir is not None:
        train_dict["data_dir"] = data_dir

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
        train_dict["save_dir"] = save_dir

    if num_workers is not None:
        train_dict["num_workers"] = num_workers

    if weight_download_region is not None:
        train_dict["weight_download_region"] = weight_download_region

    if skip_load_validation is not None:
        train_dict["skip_load_validation"] = skip_load_validation

    try:
        manager = ModelManager(
            ModelConfig(
                video_loader_config=video_loader_config,
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
    Data directory: {data_dir if data_dir is not None else config_dict["train_config"].get("data_dir")}
    Labels csv: {labels if labels is not None else config_dict["train_config"].get("labels")}
    Species: {species}
    Model name: {config.train_config.model_name}
    Checkpoint: {checkpoint if checkpoint is not None else config_dict["train_config"].get("checkpoint")}
    Batch size: {config.train_config.batch_size}
    Number of workers: {config.train_config.num_workers}
    GPUs: {config.train_config.gpus}
    Dry run: {config.train_config.dry_run}
    Save directory: {config.train_config.save_dir}
    Weight download region: {config.train_config.weight_download_region}
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
        help="Whether to save out predictions. If you want to specify the output directory, use save_dir instead.",
    ),
    save_dir: Path = typer.Option(
        None,
        help="An optional directory in which to save the model predictions and configuration yaml. "
        "Defaults to the current working directory if save is True.",
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
    skip_load_validation: bool = typer.Option(
        None,
        help="Skip check that verifies all videos can be loaded prior to inference. Only use if you're very confident all your videos can be loaded.",
    ),
    overwrite: bool = typer.Option(
        None, "--overwrite", "-o", help="Overwrite outputs in the save directory if they exist."
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
        with (MODELS_DIRECTORY / f"{model.value}/config.yaml").open() as f:
            config_dict = yaml.safe_load(f)
        config_file = None

    if "video_loader_config" in config_dict.keys():
        video_loader_config = VideoLoaderConfig(**config_dict["video_loader_config"])
    else:
        video_loader_config = None

    predict_dict = config_dict["predict_config"]

    # override if any command line arguments are passed
    if data_dir is not None:
        predict_dict["data_dir"] = data_dir

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

    # save_dir takes precedence over save
    if save_dir is not None:
        predict_dict["save_dir"] = save_dir

    if proba_threshold is not None:
        predict_dict["proba_threshold"] = proba_threshold

    if output_class_names is not None:
        predict_dict["output_class_names"] = output_class_names

    if num_workers is not None:
        predict_dict["num_workers"] = num_workers

    if weight_download_region is not None:
        predict_dict["weight_download_region"] = weight_download_region

    if skip_load_validation is not None:
        predict_dict["skip_load_validation"] = skip_load_validation

    if overwrite is not None:
        predict_dict["overwrite"] = overwrite

    try:
        manager = ModelManager(
            ModelConfig(
                video_loader_config=video_loader_config,
                predict_config=PredictConfig(**predict_dict),
            )
        )
    except ValidationError as ex:
        logger.error("Invalid configuration.")
        raise typer.Exit(ex)

    config = manager.config

    msg = f"""The following configuration will be used for inference:

    Config file: {config_file}
    Data directory: {data_dir if data_dir is not None else config_dict["predict_config"].get("data_dir")}
    Filepath csv: {filepaths if filepaths is not None else config_dict["predict_config"].get("filepaths")}
    Model: {config.predict_config.model_name}
    Checkpoint: {checkpoint if checkpoint is not None else config_dict["predict_config"].get("checkpoint")}
    Batch size: {config.predict_config.batch_size}
    Number of workers: {config.predict_config.num_workers}
    GPUs: {config.predict_config.gpus}
    Dry run: {config.predict_config.dry_run}
    Save directory: {config.predict_config.save_dir}
    Proba threshold: {config.predict_config.proba_threshold}
    Output class names: {config.predict_config.output_class_names}
    Weight download region: {config.predict_config.weight_download_region}
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


def version_callback(version: bool):
    """Print zamba package version and exit."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show zamba version and exit.",
    ),
):
    """Zamba is a tool built in Python to automatically identify the species seen
    in camera trap videos from sites in Africa and Europe. Visit
    https://zamba.drivendata.org/docs for more in-depth documentation."""


@app.command()
def densepose(
    data_dir: Path = typer.Option(
        None, exists=True, help="Path to video or image file or folder containing images/videos."
    ),
    filepaths: Path = typer.Option(
        None, exists=True, help="Path to csv containing `filepath` column with videos."
    ),
    save_dir: Path = typer.Option(
        None,
        help="An optional directory for saving the output. Defaults to the current working directory.",
    ),
    config: Path = typer.Option(
        None,
        exists=True,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    fps: float = typer.Option(
        1.0, help="Number of frames per second to process. Defaults to 1.0 (1 frame per second)."
    ),
    output_type: DensePoseOutputEnum = typer.Option(
        "chimp_anatomy",
        help="If 'chimp_anatomy' will apply anatomy model from densepose to the rendering and create a CSV with "
        "the anatomy visible in each frame. If 'segmentation', will just output the segmented area where an animal "
        "is identified, which works for more species than chimpanzees.",
    ),
    render_output: bool = typer.Option(
        False,
        help="If True, generate an output image or video with either the segmentation or anatomy rendered "
        "depending on the `output_type` that is chosen.",
    ),
    weight_download_region: RegionEnum = typer.Option(
        None, help="Server region for downloading weights."
    ),
    cache_dir: Path = typer.Option(
        None,
        exists=False,
        help="Path to directory for model weights. Alternatively, specify with environment variable `MODEL_CACHE_DIR`. If not specified, user's cache directory is used.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation of configuration and proceed right to prediction.",
    ),
):
    """Run densepose algorithm on videos.

    If an argument is specified in both the command line and in a yaml file, the command line input will take precedence.
    """
    if config is not None:
        with config.open() as f:
            config_dict = yaml.safe_load(f)
        config_file = config
    else:
        config_dict = {}
        config_file = None

    if "video_loader_config" not in config_dict.keys():
        config_dict["video_loader_config"] = dict()

    if fps is not None:
        config_dict["video_loader_config"]["fps"] = fps

    predict_dict = config_dict

    # override if any command line arguments are passed
    if data_dir is not None:
        predict_dict["data_dir"] = data_dir

    if filepaths is not None:
        predict_dict["filepaths"] = filepaths

    if save_dir is not None:
        predict_dict["save_dir"] = save_dir

    if weight_download_region is not None:
        predict_dict["weight_download_region"] = weight_download_region

    if cache_dir is not None:
        predict_dict["cache_dir"] = cache_dir

    if output_type is not None:
        predict_dict["output_type"] = output_type

    if render_output is not None:
        predict_dict["render_output"] = render_output

    try:
        densepose_config = DensePoseConfig(**predict_dict)
    except ValidationError as ex:
        logger.error("Invalid configuration.")
        raise typer.Exit(ex)

    msg = f"""The following configuration will be used for inference:

    Config file: {config_file}
    Output type: {densepose_config.output_type}
    Render output: {densepose_config.render_output}
    Data directory: {data_dir if data_dir is not None else config_dict.get("data_dir")}
    Filepath csv: {filepaths if filepaths is not None else config_dict.get("filepaths")}
    Weight download region: {densepose_config.weight_download_region}
    Cache directory: {densepose_config.cache_dir}
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            densepose_config.run_model()


@app.command()
def depth(
    filepaths: Path = typer.Option(
        None, exists=True, help="Path to csv containing `filepath` column with videos."
    ),
    data_dir: Path = typer.Option(None, exists=True, help="Path to folder containing videos."),
    save_to: Path = typer.Option(
        None,
        help="An optional directory or csv path for saving the output. Defaults to `depth_predictions.csv` in the working directory.",
    ),
    overwrite: bool = typer.Option(
        None, "--overwrite", "-o", help="Overwrite output csv if it exists."
    ),
    batch_size: int = typer.Option(None, help="Batch size to use for inference."),
    num_workers: int = typer.Option(
        None,
        help="Number of subprocesses to use for data loading.",
    ),
    gpus: int = typer.Option(
        None,
        help="Number of GPUs to use for inference. If not specifiied, will use all GPUs found on machine.",
    ),
    model_cache_dir: Path = typer.Option(
        None,
        exists=False,
        help="Path to directory for downloading model weights. Alternatively, specify with environment variable `MODEL_CACHE_DIR`. If not specified, user's cache directory is used.",
    ),
    weight_download_region: RegionEnum = typer.Option(
        None, help="Server region for downloading weights."
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation of configuration and proceed right to prediction.",
    ),
):
    """Estimate animal distance at each second in the video."""
    predict_dict = dict(filepaths=filepaths)

    # override if any command line arguments are passed
    if data_dir is not None:
        predict_dict["data_dir"] = data_dir
    if save_to is not None:
        predict_dict["save_to"] = save_to
    if overwrite is not None:
        predict_dict["overwrite"] = overwrite
    if batch_size is not None:
        predict_dict["batch_size"] = batch_size
    if num_workers is not None:
        predict_dict["num_workers"] = num_workers
    if gpus is not None:
        predict_dict["gpus"] = gpus
    if model_cache_dir is not None:
        predict_dict["model_cache_dir"] = model_cache_dir
    if weight_download_region is not None:
        predict_dict["weight_download_region"] = weight_download_region

    try:
        depth_config = DepthEstimationConfig(**predict_dict)
    except ValidationError as ex:
        logger.error("Invalid configuration.")
        raise typer.Exit(ex)

    msg = f"""The following configuration will be used for inference:

    Filepath csv: {predict_dict["filepaths"]}
    Data directory: {depth_config.data_dir}
    Save to: {depth_config.save_to}
    Overwrite: {depth_config.overwrite}
    Batch size: {depth_config.batch_size}
    Number of workers: {depth_config.num_workers}
    GPUs: {depth_config.gpus}
    Model cache: {depth_config.model_cache_dir}
    Weight download region: {depth_config.weight_download_region}
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
        depth_config.run_model()


if __name__ == "__main__":
    app()
