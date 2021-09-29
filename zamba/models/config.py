from enum import Enum
from multiprocessing import cpu_count
import os
from pathlib import Path
import random
from typing import Dict, Optional, Union

import appdirs
import ffmpeg
from loguru import logger
import pandas as pd
from pydantic import BaseModel
from pydantic import DirectoryPath, FilePath, validator, root_validator
import torch
from tqdm import tqdm
import yaml

from zamba.data.metadata import create_site_specific_splits
from zamba.data.video import VideoLoaderConfig
from zamba.exceptions import ZambaFfmpegException
from zamba.models.slowfast_models import SlowFast
from zamba.models.efficientnet_models import TimeDistributedEfficientNet
from zamba.models.utils import RegionEnum
from zamba.pytorch.transforms import zamba_image_model_transforms, slowfast_transforms
from zamba.settings import SPLIT_SEED, VIDEO_SUFFIXES, ROOT_DIRECTORY


GPUS_AVAILABLE = torch.cuda.device_count()

MODEL_MAPPING = {
    "time_distributed": {
        "full_name": "time_distributed_efficientnet_multilayer_head_mdlite",
        "model_class": TimeDistributedEfficientNet,
        "public_weights": "zamba_time_distributed_v2.ckpt",
        "private_weights": "s3://drivendata-client-zamba/data/results/time_distributed_efficientnet_multilayer_head_mdlite/version_1/checkpoints/epoch=15-step=128720-v4_zamba.ckpt",
        "config": ROOT_DIRECTORY / "zamba/models/configs/time_distributed.yaml",
        "transform": zamba_image_model_transforms(),
    },
    "european": {
        "full_name": "time_distributed_efficientnet_finetuned_european",
        "model_class": TimeDistributedEfficientNet,
        "public_weights": "zamba_european_v2.ckpt",
        "private_weights": "s3://drivendata-client-zamba/data/results/time_distributed_efficientnet_finetuned_european/version_1/checkpoints/epoch=4-step=2820-v3_zamba.ckpt",
        "config": ROOT_DIRECTORY / "zamba/models/configs/european.yaml",
        "transform": zamba_image_model_transforms(),
    },
    "slowfast": {
        "full_name": "slowfast_zamba_finetune_mdlite",
        "model_class": SlowFast,
        "public_weights": "zamba_slowfast_v2.ckpt",
        "private_weights": "s3://drivendata-client-zamba/data/results/slowfast_zamba_finetune_mdlite/version_0/checkpoints/epoch=9-step=20120-v5_zamba.ckpt",
        "config": ROOT_DIRECTORY / "zamba/models/configs/slowfast.yaml",
        "transform": slowfast_transforms(),
    },
}


class ModelEnum(str, Enum):
    """Shorthand names of models supported by zamba."""

    time_distributed = "time_distributed"
    slowfast = "slowfast"
    european = "european"


class MonitorEnum(str, Enum):
    """Validation metric to monitor for early stopping. Training is stopped when no
    improvement is observed."""

    val_macro_f1 = "val_macro_f1"
    val_loss = "val_loss"


def validate_gpus(gpus: int):
    """Ensure the number of GPUs requested is equal to or less than the number of GPUs
    available on the machine."""
    if gpus > GPUS_AVAILABLE:
        raise ValueError(f"Found only {GPUS_AVAILABLE} GPU(s). Cannot use {gpus}.")
    else:
        return gpus


def validate_cache_dir(cache_dir: Optional[Path]):
    """Set up cache directory for downloading model weight. Order of priority is:
    config argument, environment variable, or user's default cache dir.
    """
    if cache_dir is None:
        cache_dir = os.getenv("ZAMBA_CACHE_DIR", Path(appdirs.user_cache_dir()) / "zamba")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def check_files_exist_and_load(
    df: pd.DataFrame, data_directory: DirectoryPath, skip_load_validation: bool
):
    """Check whether files in file list exist and can be loaded with ffmpeg.
    Warn and skip files that don't exist or can't be loaded.

    Args:
        df (pd.DataFrame): DataFrame with a "filepath" column
        data_directory (Path): Data folder to prepend if filepath is not an
            absolute path.
        skip_load_validation (bool): Skip ffprobe check that verifies all videos
            can be loaded.

    Returns:
        pd.DataFrame: DataFrame with valid and loadable videos.
    """
    # update filepath column to prepend data_dir if filepath column is not an absolute path
    data_dir = Path(data_directory).resolve()
    df["filepath"] = str(data_dir) / df.filepath.path

    # we can have multiple rows per file with labels so limit just to one row per file for these checks
    files_df = df[["filepath"]].drop_duplicates()

    # check data exists
    logger.info(
        f"Checking all {len(files_df):,} filepaths exist. Can take up to a minute for every couple thousand files."
    )
    invalid_files = files_df[~files_df.filepath.path.exists()]

    # if no files exist
    if len(invalid_files) == len(files_df):
        raise ValueError(
            f"None of the video filepaths exist. Are you sure they're specified correctly? Here's an example invalid path: {invalid_files.filepath.values[0]}. Either specify absolute filepaths in the csv or provide filepaths relative to `data_directory`."
        )

    # if at least some files exist
    elif len(invalid_files) > 0:
        logger.debug(
            f"The following files could not be found: {'/n'.join(invalid_files.filepath.values.tolist())}"
        )
        logger.warning(
            f"Skipping {len(invalid_files)} file(s) that could not be found. For example, {invalid_files.filepath.values[0]}."
        )
        # remove invalid files to prep for ffprobe check on remaining
        files_df = files_df[~files_df.filepath.isin(invalid_files.filepath)]

    bad_load = []
    if not skip_load_validation:

        logger.info(
            "Checking that all videos can be loaded. If you're very confident all your videos can be loaded, you can skip this with `skip_load_validation`, but it's not recommended."
        )

        # ffprobe check
        for f in tqdm(files_df.filepath):
            try:
                ffmpeg.probe(str(f))
            except ffmpeg.Error as exc:
                logger.debug(ZambaFfmpegException(exc.stderr))
                bad_load.append(f)

        if len(bad_load) > 0:
            logger.warning(
                f"Skipping {len(bad_load)} file(s) that could not be loaded with ffmpeg."
            )

    df = df[
        (~df.filepath.isin(bad_load)) & (~df.filepath.isin(invalid_files.filepath))
    ].reset_index(drop=True)

    return df


def validate_model_name_and_checkpoint(cls, values):
    """Ensures a checkpoint file or model name is provided. If a model name is provided,
    looks up the corresponding checkpoint file from MODEL_MAPPING.
    """
    checkpoint = values.get("checkpoint")
    model_name = values.get("model_name")

    # must specify either checkpoint or model name
    if checkpoint is None and model_name is None:
        raise ValueError("Must provide either model_name or checkpoint path.")

    # log a warning if user specifies both
    elif checkpoint is not None and model_name is not None:
        logger.info(
            f"Both model_name and checkpoint were specified. Using checkpoint file: {checkpoint}."
        )

    elif checkpoint is None and model_name is not None:
        # look up public weights file based on model name
        values["checkpoint"] = MODEL_MAPPING[model_name]["public_weights"]

    return values


class ZambaBaseModel(BaseModel):
    """Set defaults for all models that inherit from the pydantic base model."""

    class Config:
        extra = "forbid"
        use_enum_values = True
        validate_assignment = True


class BackboneFinetuneConfig(ZambaBaseModel):
    """Configuration containing parameters to be used for backbone finetuning.

    Args:
        unfreeze_backbone_at_epoch (int, optional): Epoch at which the backbone
            will be unfrozen. Defaults to 15.
        backbone_initial_ratio_lr (float, optional): Used to scale down the backbone
            learning rate compared to rest of model. Defaults to 0.01.
        multiplier (int or float, optional): Multiply the learning rate by a constant
            value at the end of each epoch. Defaults to 1.
        pre_train_bn (bool, optional): Train batch normalization layers prior to
            finetuning. False is recommended for slowfast models and True is recommended
            for time distributed models. Defaults to False.
        train_bn (bool, optional): Make batch normalization trainable. Defaults to False.
        verbose (bool, optional): Display current learning rate for model and backbone.
            Defaults to True.
    """

    unfreeze_backbone_at_epoch: Optional[int] = 15
    backbone_initial_ratio_lr: Optional[float] = 0.01
    multiplier: Optional[Union[int, float]] = 1
    pre_train_bn: Optional[bool] = False  # freeze batch norm layers prior to finetuning
    train_bn: Optional[bool] = False  # don't train bn layers in unfrozen finetuning layers
    verbose: Optional[bool] = True


class EarlyStoppingConfig(ZambaBaseModel):
    """Configuration containing parameters to be used for early stopping.

    Args:
        monitor (str): Metric to be monitored. Options are "val_macro_f1" or
            "val_loss". Defaults to "val_macro_f1".
        patience (int): Number of epochs with no improvement after which training
            will be stopped. Defaults to 3.
        verbose (bool): Verbosity mode. Defaults to True.
        mode (str, optional): Options are "min" or "max". In "min" mode, training
            will stop when the quantity monitored has stopped decreasing and in
            "max" mode it will stop when the quantity monitored has stopped increasing.
            If None, mode will be inferred from monitor. Defaults to None.
    """

    monitor: MonitorEnum = "val_macro_f1"
    patience: int = 3
    verbose: bool = True
    mode: Optional[str] = None

    @root_validator()
    def validate_mode(cls, values):
        mode = {"val_macro_f1": "max", "val_loss": "min"}[values.get("monitor")]
        user_mode = values.get("mode")
        if user_mode is None:
            values["mode"] = mode
        elif user_mode != mode:
            raise ValueError(
                f"Provided mode {user_mode} is incorrect for {values.get('monitor')} monitor."
            )
        return values


class SchedulerConfig(ZambaBaseModel):
    """Configuration containing parameters for a custom pytorch learning rate scheduler.
    See https://pytorch.org/docs/stable/optim.html for options.

    Args:
        scheduler (str): Name of learning rate scheduler to use. See
            https://pytorch.org/docs/stable/optim.html for options.
        scheduler_params (dict, optional): Parameters passed to learning rate
            scheduler upon initialization (eg. {"milestones": [1], "gamma": 0.5,
            "verbose": True}). Defaults to None.
    """

    scheduler: Optional[str]
    scheduler_params: Optional[dict] = None

    @validator("scheduler", always=True)
    def validate_scheduler(cls, scheduler):
        if scheduler is None:
            return None

        elif scheduler not in torch.optim.lr_scheduler.__dict__.keys():
            raise ValueError(
                "Scheduler is not a `torch.optim.lr_scheduler`. "
                "See https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py "
                "for options."
            )
        else:
            return scheduler


class TrainConfig(ZambaBaseModel):
    """
    Configuration for training a model.

    Args:
        labels (FilePath or pandas DataFrame): Path to a CSV or pandas DataFrame
            containing labels for training, with one row per label. There must be
            columns called 'filepath' (absolute or relative to the data_directory) and
            'label', and optionally columns called 'split' ("train", "val", or "holdout")
            and 'site'. Labels must be specified to train a model.
        data_directory (DirectoryPath): Path to a directory containing training
            videos. Defaults to the working directory.
        model_name (str, optional): Name of the model to use for training. Options
            are: time_distributed, slowfast, european. Defaults to time_distributed.
        checkpoint (FilePath, optional): Path to a custom checkpoint file (.ckpt)
            generated by zamba that can be used to resume training. If None, defaults
            to a pretrained model. Defaults to None.
        scheduler_config (SchedulerConfig or str, optional): Config for setting up
            the learning rate scheduler on the model. If "default", uses scheduler
            that was used for training. If None, will not use a scheduler.
            Defaults to "default".
        dry_run (bool or int, Optional): Run one training and validation batch
            for one epoch to detect any bugs prior to training the full model.
            Disables tuners, checkpoint callbacks, loggers, and logger callbacks.
            Defaults to False.
        batch_size (int): Batch size to use for training. Defaults to 8.
        auto_lr_find (bool): Use a learning rate finder algorithm when calling
            trainer.tune() to find a optimal initial learning rate. Defaults to True.
        backbone_finetune_params (BackboneFinetuneConfig, optional): Set parameters
            to finetune a backbone model to align with the current learning rate.
            Defaults to a BackboneFinetuneConfig(unfreeze_backbone_at_epoch=15,
            backbone_initial_ratio_lr=0.01, multiplier=1, pre_train_bn=False,
            train_bn=False, verbose=True).
        gpus (int): Number of GPUs to train on applied per node.
            Defaults to all of the available GPUs found on the machine.
        num_workers (int): Number of subprocesses to use for data loading. 0 means
            that the data will be loaded in the main process. Defauts to one less
            than the number of CPUs in the system, or 1 if there is one CPU in the
            system.
        max_epochs (int, optional): Stop training once this number of epochs is
            reached. Disabled by default (None), which stops training at 1000 epochs.
        early_stopping_config (EarlyStoppingConfig, optional): Configuration for
            early stopping, which monitors a metric during training and stops training
            when the metric stops improving. Defaults to EarlyStoppingConfig(monitor='val_macro_f1',
            patience=3, verbose=True, mode='max').
        weight_download_region (str): s3 region to download pretrained weights from.
            Options are "us" (United States), "eu" (European Union), or "asia"
            (Asia Pacific). Defaults to "us".
        cache_dir (Path, optional): Cache directory where downloaded model weights
            will be saved. If None and the ZAMBA_CACHE_DIR environment variable is
            not set, uses your default cache directory. Defaults to None.
        split_proportions (dict): Proportions used to divide data into training,
            validation, and holdout sets if a if a "split" column is not included in
            labels. Defaults to "train": 3, "val": 1, "holdout": 1.
        save_directory (Path, optional): Path to a directory where training files
            will be saved. Files include the best model checkpoint (``model_name``.ckpt),
            training configuration (configuration.yaml), Tensorboard logs
            (events.out.tfevents...), test metrics (test_metrics.json), validation
            metrics (val_metrics.json), and model hyperparameters (hparams.yml).
            If None, a folder is created in the working directory called
            "zamba_``model_name``". Defaults to None.
        overwrite_save_directory (bool): If True, will save outputs in `save_directory`
            overwriting if those exist. If False, will create auto-incremented `version_n` folder
            in `save_directory` with model outputs. Defaults to False.
        skip_load_validation (bool). Skip ffprobe check, which verifies that all
            videos can be loaded and skips files that cannot be loaded. Defaults
            to False.
        from_scratch (bool): Instantiate the model with base weights. This means
            starting with ImageNet weights for image-based models (time_distributed
            and european) and Kinetics weights for video-based models (slowfast).
            Defaults to False.
        predict_all_zamba_species (bool): Output all zamba species rather than
            only the species in the labels file.
    """

    labels: Union[FilePath, pd.DataFrame]
    data_directory: DirectoryPath = Path.cwd()
    checkpoint: Optional[FilePath] = None
    scheduler_config: Optional[Union[str, SchedulerConfig]] = "default"
    model_name: Optional[ModelEnum] = ModelEnum.time_distributed
    dry_run: Union[bool, int] = False
    batch_size: int = 8
    auto_lr_find: bool = True
    backbone_finetune_config: Optional[BackboneFinetuneConfig] = BackboneFinetuneConfig()
    gpus: int = GPUS_AVAILABLE
    num_workers: int = max(cpu_count() - 1, 1)
    max_epochs: Optional[int] = None
    early_stopping_config: Optional[EarlyStoppingConfig] = EarlyStoppingConfig()
    weight_download_region: RegionEnum = "us"
    cache_dir: Optional[Path] = None
    split_proportions: Optional[Dict[str, int]] = {"train": 3, "val": 1, "holdout": 1}
    save_directory: Path = Path.cwd()
    overwrite_save_directory: bool = False
    skip_load_validation: bool = False
    from_scratch: bool = False
    predict_all_zamba_species: bool = True

    class Config:
        arbitrary_types_allowed = True

    _validate_gpus = validator("gpus", allow_reuse=True, pre=True)(validate_gpus)

    _validate_cache_dir = validator("cache_dir", allow_reuse=True, always=True)(validate_cache_dir)

    @root_validator(skip_on_failure=True)
    def validate_from_scratch_and_checkpoint(cls, values):
        if values["from_scratch"]:
            if values["checkpoint"] is not None:
                raise ValueError("If from_scratch=True, you cannot specify a checkpoint.")

            if values["model_name"] is None:
                raise ValueError("If from_scratch=True, model_name cannot be None.")

        return values

    _validate_model_name_and_checkpoint = root_validator(allow_reuse=True)(
        validate_model_name_and_checkpoint
    )

    @validator("scheduler_config", always=True)
    def validate_scheduler_config(cls, scheduler_config):
        if scheduler_config is None:
            return SchedulerConfig(scheduler=None)
        elif isinstance(scheduler_config, str) and scheduler_config != "default":
            raise ValueError("Scheduler can either be 'default', None, or a SchedulerConfig.")
        else:
            return scheduler_config

    @root_validator(skip_on_failure=True)
    def turn_off_load_validation_if_dry_run(cls, values):
        if values["dry_run"] and not values["skip_load_validation"]:
            logger.info("Turning off video loading check since dry_run=True.")
            values["skip_load_validation"] = True
        return values

    @root_validator(skip_on_failure=True)
    def validate_filepaths_and_labels(cls, values):
        logger.info("Validating labels csv.")
        labels = (
            pd.read_csv(values["labels"])
            if not isinstance(values["labels"], pd.DataFrame)
            else values["labels"]
        )

        if not set(["label", "filepath"]).issubset(labels.columns):
            raise ValueError(f"{values['labels']} must contain `filepath` and `label` columns.")

        # validate split column has no partial nulls or invalid values
        if "split" in labels.columns:

            # if split is entirely null, warn, drop column, and generate splits automatically
            if labels.split.isnull().all():
                logger.warning(
                    "Split column is entirely null. Will generate splits automatically using `split_proportions`."
                )
                labels = labels.drop("split", axis=1)

            # error if split column has null values
            elif labels.split.isnull().any():
                raise ValueError(
                    f"Found {labels.split.isnull().sum()} row(s) with null `split`. Fill in these rows with either `train`, `val`, or `holdout`. Alternatively, do not include a `split` column in your labels and we'll generate splits for you using `split_proportions`."
                )

            # otherwise check that split values are valid
            elif not set(labels.split).issubset({"train", "val", "holdout"}):
                raise ValueError(
                    f"Found the following invalid values for `split`: {set(labels.split).difference({'train', 'val', 'holdout'})}. `split` can only contain `train`, `val`, or `holdout.`"
                )

            elif values["split_proportions"] is not None:
                logger.warning(
                    "Labels contains split column yet split_proprtions are also provided. Split column in labels takes precendece."
                )

        # error if labels are entirely null
        null_labels = labels.label.isnull()
        if sum(null_labels) == len(labels):
            raise ValueError("Species cannot be null for all videos.")

        # skip and warn about any videos without species label
        elif sum(null_labels) > 0:
            logger.warning(f"Found {sum(null_labels)} filepath(s) with no label. Will skip.")
            labels = labels[~null_labels]

        # check that all videos exist and can be loaded
        values["labels"] = check_files_exist_and_load(
            df=labels,
            data_directory=values["data_directory"],
            skip_load_validation=values["skip_load_validation"],
        )
        return values

    @root_validator(skip_on_failure=True)
    def preprocess_labels(cls, values):
        logger.info("Preprocessing labels into one hot encoded labels with one row per video.")
        # one hot encode collapse to one row per video
        labels = (
            pd.get_dummies(
                values["labels"].rename(columns={"label": "species"}), columns=["species"]
            )
            .groupby("filepath")
            .max()
        )

        # if no "split" column, set up train, val, and holdout split
        if "split" not in labels.columns:
            logger.info(
                f"Dividing videos into train, val, and holdout sets using the following split proportions: {values['split_proportions']}."
            )

            # use site info if we have it
            if "site" in labels.columns:
                logger.info("Using provided 'site' column to do a site-specific split")
                labels["split"] = create_site_specific_splits(
                    labels["site"], proportions=values["split_proportions"]
                )
            else:
                logger.info(
                    "No 'site' column found so videos will be randomly allocated to splits."
                )
                # otherwise randomly allocate
                random.seed(SPLIT_SEED)
                labels["split"] = random.choices(
                    list(values["split_proportions"].keys()),
                    weights=list(values["split_proportions"].values()),
                    k=len(labels),
                )

        # filepath becomes column instead of index
        values["labels"] = labels.reset_index()
        return values


class PredictConfig(ZambaBaseModel):
    """
    Configuration for using a model for inference.

    Args:
        filepaths (FilePath): Path to a CSV containing videos for inference, with
            one row per video in the data_directory. There must be a column called
            'filepath' (absolute or relative to the data_directory). If None, uses
            all files in data_directory. Defaults to None.
        data_directory (DirectoryPath): Path to a directory containing videos for
            inference. Defaults to the working directory.
        model_name (str, optional): Name of the model to use for inference. Options
            are: time_distributed, slowfast, european. Defaults to time_distributed.
        checkpoint (FilePath, optional): Path to a custom checkpoint file (.ckpt)
            generated by zamba that can be used to generate predictions. If None,
            defaults to a pretrained model. Defaults to None.
        gpus (int): Number of GPUs to use for inference.
            Defaults to all of the available GPUs found on the machine.
        num_workers (int): Number of subprocesses to use for data loading. 0 means
            that the data will be loaded in the main process. Defauts to one less
            than the number of CPUs in the system, or 1 if there is one CPU in the
            system.
        batch_size (int): Batch size to use for inference. Defaults to 8.
        save (bool or Path): Path to a CSV to save predictions. If True is passed,
            "zamba_predictions.csv" is written to the current working directory.
            If False is passed, predictions are not saved. Defaults to True.
        dry_run (bool): Perform inference on a single batch for testing. Predictions
            will not be saved. Defaults to False.
        proba_threshold (float, optional): Probability threshold for classification.
            If specified, binary predictions are returned with 1 being greater than the
            threshold and 0 being less than or equal to the threshold. If None, return
            probability scores for each species. Defaults to None.
        output_class_names (bool): Output the species with the highest probability
            score as a single prediction for each video. If False, return probabilty
            scores for each species. Defaults to False.
        weight_download_region (str): s3 region to download pretrained weights from.
            Options are "us" (United States), "eu" (European Union), or "asia"
            (Asia Pacific). Defaults to "us".
        cache_dir (Path, optional): Cache directory where downloaded model weights
            will be saved. If None and no environment variable is set, will use your
            default cache directory. Defaults to None.
        skip_load_validation (bool). By default, zamba runs a check to verify that
            all videos can be loaded and skips files that cannot be loaded. This can
            be time intensive, depending on how many videos there are. If you are very
            confident all your videos can be loaded, you can set this to True and skip
            this check. Defaults to False.
    """

    data_directory: DirectoryPath = Path.cwd()
    filepaths: Optional[FilePath] = None
    checkpoint: Optional[FilePath] = None
    model_name: Optional[ModelEnum] = ModelEnum.time_distributed
    gpus: int = GPUS_AVAILABLE
    num_workers: int = max(cpu_count() - 1, 1)
    batch_size: int = 8
    save: Union[bool, Path] = True
    dry_run: bool = False
    proba_threshold: Optional[float] = None
    output_class_names: bool = False
    weight_download_region: RegionEnum = "us"
    cache_dir: Optional[Path] = None
    skip_load_validation: bool = False

    _validate_gpus = validator("gpus", allow_reuse=True, pre=True)(validate_gpus)

    _validate_cache_dir = validator("cache_dir", allow_reuse=True, always=True)(validate_cache_dir)

    @root_validator(skip_on_failure=True)
    def validate_dry_run_and_save(cls, values):
        if values["dry_run"] and (values["save"] is not False):
            logger.warning("Cannot save when predicting with dry_run=True. Setting save=False.")
            values["save"] = False

        return values

    @root_validator(skip_on_failure=True)
    def validate_save(cls, values):
        # do this check before we look up checkpoints based on model name so we can see if checkpoint is None
        save = values["save"]
        checkpoint = values["checkpoint"]

        # if False, no predictions will be written out
        if save is False:
            return values

        else:
            # if save=True and we have a local checkpoint, save in checkpoint directory
            if save is True and checkpoint is not None:
                save = checkpoint.parent / "zamba_predictions.csv"

            # else, save to current working directory
            elif save is True and checkpoint is None:
                save = Path.cwd() / "zamba_predictions.csv"

        # validate save path
        if isinstance(save, Path):
            if save.suffix.lower() != ".csv":
                raise ValueError("Save path must end with .csv")
            elif save.exists():
                raise ValueError(f"Save path {save} already exists.")
            else:
                values["save"] = save

            # create any needed parent directories
            save.parent.mkdir(parents=True, exist_ok=True)

        return values

    _validate_model_name_and_checkpoint = root_validator(allow_reuse=True)(
        validate_model_name_and_checkpoint
    )

    @root_validator
    def validate_proba_threshold(cls, values):
        if values["proba_threshold"] is not None:
            if (values["proba_threshold"] <= 0) or (values["proba_threshold"] >= 1):
                raise ValueError(
                    "Setting proba_threshold outside of the range (0, 1) will cause all probabilities to be rounded to the same value."
                )

            if values["output_class_names"] is True:
                logger.warning(
                    "`output_class_names` will be ignored because `proba_threshold` is specified."
                )
        return values

    @root_validator(pre=False)
    def get_filepaths(cls, values):
        """If no file list is passed, get all files in data directory. Warn if there
        are unsupported suffixes. Filepaths is set to a dataframe, where column `filepath`
        contains files with valid suffixes.
        """
        if values["filepaths"] is None:
            logger.info(f"Getting files in {values['data_directory']}.")
            files = []
            new_suffixes = []

            # iterate over all files in data directory
            for f in values["data_directory"].rglob("*"):
                if f.is_file():
                    # keep just files with supported suffixes
                    if f.suffix.lower() in VIDEO_SUFFIXES:
                        files.append(f.resolve())
                    else:
                        new_suffixes.append(f.suffix.lower())

            if len(new_suffixes) > 0:
                logger.warning(
                    f"Ignoring {len(new_suffixes)} file(s) with suffixes {set(new_suffixes)}. To include, specify all video suffixes with a VIDEO_SUFFIXES environment variable."
                )

            if len(files) == 0:
                raise ValueError(f"No video files found in {values['data_directory']}.")

            logger.info(f"Found {len(files)} videos in {values['data_directory']}.")
            values["filepaths"] = pd.DataFrame(files, columns=["filepath"])
        return values

    @root_validator(skip_on_failure=True)
    def validate_files(cls, values):
        # if globbing from data directory, already have valid dataframe
        if isinstance(values["filepaths"], pd.DataFrame):
            files_df = values["filepaths"]
        else:
            # make into dataframe even if only one column for clearer indexing
            files_df = pd.DataFrame(pd.read_csv(values["filepaths"]))

        if "filepath" not in files_df.columns:
            raise ValueError(f"{values['filepaths']} must contain a `filepath` column.")

        # can only contain one row per filepath
        num_duplicates = len(files_df) - files_df.filepath.nunique()
        if num_duplicates > 0:
            logger.warning(
                f"Found {num_duplicates} duplicate row(s) in filepaths csv. Dropping duplicates so predictions will have one row per video."
            )

        values["filepaths"] = check_files_exist_and_load(
            df=files_df,
            data_directory=values["data_directory"],
            skip_load_validation=values["skip_load_validation"],
        )
        return values


class ModelConfig(ZambaBaseModel):
    """Contains all configs necessary to use a model for training or inference.
    Must contain a train_config or a predict_config at a minimum.

    Args:
        video_loader_config (VideoLoaderConfig, optional): An instantiated VideoLoaderConfig.
            Defaults to None.
        train_config (TrainConfig, optional): An instantiated TrainConfig.
            Defaults to None.
        predict_config (PredictConfig, optional): An instantiated PredictConfig.
            Defaults to None.
    """

    video_loader_config: Optional[VideoLoaderConfig] = None
    train_config: Optional[TrainConfig] = None
    predict_config: Optional[PredictConfig] = None

    class Config:
        json_loads = yaml.safe_load

    @root_validator
    def one_config_must_exist(cls, values):
        if values["train_config"] is None and values["predict_config"] is None:
            raise ValueError("Must provide either `train_config` or `predict_config`.")
        else:
            return values
