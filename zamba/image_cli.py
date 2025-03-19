from collections.abc import Callable
import os
from pathlib import Path

from click.core import ParameterSource
from loguru import logger
from pydantic.error_wrappers import ValidationError
import typer
import yaml

from zamba.images.bbox import BboxInputFormat
from zamba.images.config import (
    ImageClassificationPredictConfig,
    ImageClassificationTrainingConfig,
    ImageModelEnum,
    ResultsFormat,
)
from zamba.images.manager import ZambaImagesManager
from zamba.models.utils import RegionEnum

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_cli_args(ctx: typer.Context, source=ParameterSource.COMMANDLINE) -> dict:
    """
    Returns a dictionary of arguments that were explicitly passed via CLI.
    """
    cli_args = {}

    for param_name, value in ctx.params.items():
        # Check if the parameter was provided via CLI
        param_source = ctx.get_parameter_source(param_name)
        if param_source == source:
            cli_args[param_name] = value

    return cli_args


def consolidate_options(
    ctx: typer.Context,
    config_file: Path | None,
    config_key,
    mutators: dict[str, Callable] = {},
    to_drop: list[str] = [],
) -> dict:
    """Bring together options from different sources into a single dictionary using clear precedence rules."""
    # Load the cli defaults first
    options = get_cli_args(ctx, ParameterSource.DEFAULT)

    # Drop anything that defaults to None in favor of the pydantic defaults
    keys_to_delete = [key for key, value in options.items() if value is None]
    for key in keys_to_delete:
        del options[key]

    # Then load environment variables
    options.update(get_cli_args(ctx, ParameterSource.ENVIRONMENT))

    # Then load the configuration file if provided
    if config_file is not None:
        with config_file.open() as f:
            config_dict = yaml.safe_load(f)
        options.update(config_dict[config_key])

    # Then add any CLI arguments that were explicitly passed
    cli_args = get_cli_args(ctx)

    for mututor_name, mutator in mutators.items():
        if mututor_name in cli_args:
            new_key, value = mutator(cli_args[mututor_name])
            cli_args[new_key] = value

    options.update(cli_args)

    # Drop anything that's not for the config object
    for key in to_drop:
        if key in options:
            del options[key]

    return options


@app.command("predict")
def predict(
    ctx: typer.Context,
    data_dir: Path = typer.Option(None, exists=True, help="Path to folder containing images."),
    filepaths: Path = typer.Option(
        None,
        exists=True,
        help="Path to csv containing `filepath` column with image paths.",
    ),
    model: ImageModelEnum = typer.Option(
        None,
        help="Model to use for inference. Model will be superseded by checkpoint if provided.",
    ),
    checkpoint: Path = typer.Option(
        default=None,
        exists=True,
        help="Path to model checkpoint.",
    ),
    save: bool = typer.Option(
        None,
        help="Whether to save out predictions. If you want to specify the output directory, use save_dir instead.",
    ),
    save_dir: Path = typer.Option(
        None,
        help="An optional directory or csv path for saving the output. Defaults to `.csv` file in the working directory.",
    ),
    results_file_name: Path = typer.Option(
        None,
        help="The filename for the output predictions in the save directory.",
    ),
    results_file_format: ResultsFormat = typer.Option(
        None,
        help="The format in which to output the predictions. Currently `csv` and `megadetector` JSON formats are supported.",
    ),
    config: Path | None = typer.Option(
        None,
        exists=True,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    detections_threshold: float = typer.Option(
        None,
        help="An optional threshold for detector to detect animal on image. Defaults 0.2.",
    ),
    gpus: int = typer.Option(
        None,
        help="Number of GPUs to use for inference. If not specifiied, will use all GPUs found on machine.",
    ),
    weight_download_region: RegionEnum = typer.Option(
        None, help="Server region for downloading weights."
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
    """Identify species in an image. Option defaults are visible in the ImageClassificationPredictConfig class."""

    predict_dict = consolidate_options(
        ctx,
        config_file=config,
        config_key="predict_config",
        mutators={"model": lambda x: ("model_name", x.value)},
        to_drop=["config", "yes", "batch_size", "model"],
    )

    try:
        image_config = ImageClassificationPredictConfig(**predict_dict)
    except ValidationError as ex:
        logger.error(f"Invalid configuration: {ex}")
        raise typer.Exit(1)

    msg = f"""The following configuration will be used for inference:

    Filepath csv: {predict_dict["filepaths"] if "filepaths" in predict_dict else "Not provided"}
    Data directory: {image_config.data_dir}
    Model cache: {image_config.checkpoint}
    Detections threshold: {image_config.detections_threshold}
    GPUs: {image_config.gpus}
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
        manager = ZambaImagesManager()
        manager.predict(image_config)


@app.command("train")
def train(
    ctx: typer.Context,
    data_dir: Path = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to the images directory.",
    ),
    labels: Path = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False, help="Path to the labels."
    ),
    model: ImageModelEnum = typer.Option(
        None,
        help="Model to fine-tune. Model will be superseded by checkpoint if provided.",
    ),
    model_checkpoint: Path = typer.Option(
        None,
        exists=True,
        help="Path to model checkpoint to resume training.",
    ),
    config: Path | None = typer.Option(
        None,
        exists=True,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    lr: float = typer.Option(None, help="Learning rate."),
    batch_size: int = typer.Option(
        None,
        help="Batch size to use for training. With a value of 'None', we will try to find a maximal batch size that still fits within memory.",
    ),
    accumulated_batch_size: int = typer.Option(
        None, help="Accumulated batch size; will accumulate gradients to this virtual batch size."
    ),
    max_epochs: int = typer.Option(None, help="Max training epochs."),
    cache_dir: Path = typer.Option(
        None,
        file_okay=False,
        dir_okay=True,
        help="Path to the folder where clipped images will be saved. Applies only to training with images cropping (e.g. with bbox from coco format).",
    ),
    save_dir: Path = typer.Option(
        None,
        help="An optional directory for saving the output. Defaults to the current working directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    early_stopping_patience: int = typer.Option(
        None,
        help="Number of epochs with no improvement after which training will be stopped.",
    ),
    image_size: int = typer.Option(None, help="Image size after resize."),
    no_crop_images: bool = typer.Option(
        None,
        "--no-crop-images",
        help="Pass if you have already cropped images to the bounding box of the animal.",
    ),
    extra_train_augmentations: bool = typer.Option(
        None,
        "--extra-train-augmentations",
        help="Use extra train augmentations.",
    ),
    labels_format: BboxInputFormat = typer.Option(
        None, help="Format for bounding box annotations when labels are provided as JSON."
    ),
    num_workers: int = typer.Option(
        None,
        help="Number of subprocesses to use for data loading.",
    ),
    devices: str = typer.Option(
        None,
        help="Pytorch Lightning devices parameter (number or which GPUs to use for training).",
    ),
    accelerator: str = typer.Option(
        None,
        help="Pytorch Lightning accelerator parameter; e.g. 'cpu' or 'gpu' uses GPU if available.",
    ),
    mlflow_tracking_uri: str = typer.Option(None, help="MLFlow tracking URI"),
    mlflow_experiment_name: str = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Classification experiment name (MLFlow).",
    ),
    checkpoint_path: str = typer.Option(None, help="Dir to save training checkpoints"),
    from_scratch: bool = typer.Option(
        None,
        "--from-scratch",
        help="Training from scratch.",
    ),
    weighted_loss: bool = typer.Option(
        None,
        "--weighted-loss",
        help="Use weighted cross entropy as loss.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation of configuration and proceed right to prediction.",
    ),
):
    """Train image classifier."""
    skip_confirmation = (
        os.getenv("LOCAL_RANK", 0) != 0
    )  # True if distributed training has already started (and this is second process)

    training_dict = consolidate_options(
        ctx,
        config_file=config,
        config_key="train_config",
        mutators={
            "model": lambda x: ("model_name", x.value),
            "mlflow_experiment_name": lambda x: ("name", x),
            "model_checkpoint": lambda x: ("checkpoint", x),
            "no_crop_images": lambda x: ("crop_images", not x),
        },
        to_drop=[
            "batch_size",
            "config",
            "mlflow_experiment_name",
            "model",
            "model_checkpoint",
            "no_crop_images",
            "yes",
        ],
    )

    required_args = ("data_dir", "labels")
    for required_arg in required_args:
        if training_dict[required_arg] is None:
            raise RuntimeError(f"`{required_arg}` argument is required")

    try:
        image_config = ImageClassificationTrainingConfig(**training_dict)
    except ValidationError as ex:
        logger.error(f"Invalid configuration: {ex}")
        raise typer.Exit(1)

    # Only show confirmation on main process
    if not skip_confirmation:
        msg = f"""The following configuration will be used for training:

        Filepath csv: {training_dict["labels"]}
        Data directory: {training_dict["data_dir"]}
        Base model name: {image_config.model_name}
        Cache dir: {image_config.cache_dir}
        Learning rate: {image_config.lr}
        Batch size: {image_config.batch_size}
        Accumulated batch size: {image_config.accumulated_batch_size}
        Max epochs: {image_config.max_epochs}
        Early stopping patience: {image_config.early_stopping_patience}
        Num workers: {image_config.num_workers}
        Accelerator: {image_config.accelerator}
        Devices: {image_config.devices}
        MLFlow tracking URI: {image_config.mlflow_tracking_uri}
        MLFlow classification experiment name: {image_config.name}
        Checkpoints dir: {image_config.checkpoint_path}
        Model checkpoint: {image_config.checkpoint}
        Weighted loss: {image_config.weighted_loss}
        Extra train augmentations: {image_config.extra_train_augmentations}
        """

        if yes:
            typer.echo(f"{msg}\n\nSkipping confirmation and proceeding to prediction.")
        else:
            yes = typer.confirm(
                f"{msg}\n\nIs this correct?",
                abort=False,
                default=True,
            )
    else:
        yes = True  # Non-main processes should always proceed

    if yes:
        # kick off training
        manager = ZambaImagesManager()
        manager.train(image_config)
