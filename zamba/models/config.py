from enum import Enum
import os
from pathlib import Path
import random
from typing import Dict, Optional, Union

import appdirs
import ffmpeg
from loguru import logger
import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic import DirectoryPath, FilePath, validator, root_validator
from pqdm.threads import pqdm
import torch
from tqdm import tqdm
import yaml

from zamba import MODELS_DIRECTORY
from zamba.data.metadata import create_site_specific_splits
from zamba.data.video import VideoLoaderConfig
from zamba.exceptions import ZambaFfmpegException
from zamba.models.registry import available_models
from zamba.models.utils import (
    download_weights,
    get_checkpoint_hparams,
    get_model_checkpoint_filename,
    get_model_species,
    RegionEnum,
)
from zamba.pytorch.transforms import zamba_image_model_transforms, slowfast_transforms
from zamba.settings import IMAGE_SUFFIXES, PREDICT_ON_IMAGES, SPLIT_SEED, VIDEO_SUFFIXES


GPUS_AVAILABLE = torch.cuda.device_count()

WEIGHT_LOOKUP = {
    "time_distributed": "s3://drivendata-client-zamba/data/results/zamba_classification_retraining/td_full_set/version_1/",
    "european": "s3://drivendata-client-zamba/data/results/zamba_v2_classification/european_td_dev_base/version_0/",
    "slowfast": "s3://drivendata-client-zamba/data/results/zamba_v2_classification/experiments/slowfast_small_set_full_size_mdlite/version_2/",
    "blank_nonblank": "s3://drivendata-client-zamba/data/results/zamba_classification_retraining/td_full_set_bnb/version_0/",
}

MODEL_MAPPING = {
    "TimeDistributedEfficientNet": {
        "transform": zamba_image_model_transforms(),
        "n_frames": 16,
    },
    "SlowFast": {"transform": slowfast_transforms(), "n_frames": 32},
}


class ModelEnum(str, Enum):
    """Shorthand names of models supported by zamba."""

    time_distributed = "time_distributed"
    slowfast = "slowfast"
    european = "european"
    blank_nonblank = "blank_nonblank"


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


def validate_model_cache_dir(model_cache_dir: Optional[Path]):
    """Set up cache directory for downloading model weight. Order of priority is:
    config argument, environment variable, or user's default cache dir.
    """
    if model_cache_dir is None:
        model_cache_dir = os.getenv("MODEL_CACHE_DIR", Path(appdirs.user_cache_dir()) / "zamba")

    model_cache_dir = Path(model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    return model_cache_dir


def check_files_exist_and_load(
    df: pd.DataFrame, data_dir: DirectoryPath, skip_load_validation: bool
):
    """Check whether files in file list exist and can be loaded with ffmpeg.
    Warn and skip files that don't exist or can't be loaded.

    Args:
        df (pd.DataFrame): DataFrame with a "filepath" column
        data_dir (Path): Data folder to prepend if filepath is not an
            absolute path.
        skip_load_validation (bool): Skip ffprobe check that verifies all videos
            can be loaded.

    Returns:
        pd.DataFrame: DataFrame with valid and loadable videos.
    """
    # update filepath column to prepend data_dir
    df["filepath"] = str(data_dir) / df.filepath.path

    # we can have multiple rows per file with labels so limit just to one row per file for these checks
    files_df = df[["filepath"]].drop_duplicates()

    # check for missing files
    logger.info(f"Checking all {len(files_df):,} filepaths exist. Trying fast file checking...")

    # try to check files in parallel
    paths = files_df["filepath"].apply(Path)
    exists = pqdm(paths, Path.exists, n_jobs=16)
    exists = np.array(exists)

    # if fast checking fails, fall back to slow checking
    # if an I/O error is in `exists`, the array has dtype `object`
    if exists.dtype != bool:
        logger.info(
            "Fast file checking failed. Running slower check, which can take 30 seconds per thousand files."
        )
        exists = files_df["filepath"].path.exists()

    # select the missing files
    invalid_files = files_df[~exists]

    # if no files exist
    if len(invalid_files) == len(files_df):
        raise ValueError(
            f"None of the video filepaths exist. Are you sure they're specified correctly? Here's an example invalid path: {invalid_files.filepath.values[0]}. Either specify absolute filepaths in the csv or provide filepaths relative to `data_dir`."
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
    looks up the corresponding public checkpoint file from the official configs.
    Download the checkpoint if it does not yet exist.
    """
    checkpoint = values.get("checkpoint")
    model_name = values.get("model_name")

    # must specify either checkpoint or model name
    if checkpoint is None and model_name is None:
        raise ValueError("Must provide either model_name or checkpoint path.")

    # checkpoint supercedes model
    elif checkpoint is not None and model_name is not None:
        logger.info(f"Using checkpoint file: {checkpoint}.")
        # get model name from checkpoint so it can be used for the video loader config
        hparams = get_checkpoint_hparams(checkpoint)
        values["model_name"] = available_models[hparams["model_class"]]._default_model_name

    elif checkpoint is None and model_name is not None:
        if not values.get("from_scratch"):
            # get public weights file from official models config
            values["checkpoint"] = get_model_checkpoint_filename(model_name)

            # if cached version exists, use that
            cached_path = Path(values["model_cache_dir"]) / values["checkpoint"]
            if cached_path.exists():
                values["checkpoint"] = cached_path

            # download if checkpoint doesn't exist
            if not values["checkpoint"].exists():
                logger.info(f"Downloading weights for model to {values['model_cache_dir']}.")
                values["checkpoint"] = download_weights(
                    filename=str(values["checkpoint"]),
                    weight_region=values["weight_download_region"],
                    destination_dir=values["model_cache_dir"],
                )

    return values


def get_filepaths(cls, values):
    """If no file list is passed, get all files in data directory. Warn if there
    are unsupported suffixes. Filepaths is set to a dataframe, where column `filepath`
    contains files with valid suffixes.
    """
    if values["filepaths"] is None:
        logger.info(f"Getting files in {values['data_dir']}.")
        files = []
        new_suffixes = []

        # iterate over all files in data directory
        for f in Path(values["data_dir"]).rglob("*"):
            if f.is_file():
                # keep just files with supported suffixes
                if f.suffix.lower() in VIDEO_SUFFIXES:
                    files.append(f.resolve())
                elif PREDICT_ON_IMAGES and f.suffix.lower() in IMAGE_SUFFIXES:
                    files.append(f.resolve())
                else:
                    new_suffixes.append(f.suffix.lower())

        if len(new_suffixes) > 0:
            logger.warning(
                f"Ignoring {len(new_suffixes)} file(s) with suffixes {set(new_suffixes)}. To include, specify all video suffixes with a VIDEO_SUFFIXES environment variable."
            )

        if len(files) == 0:
            raise ValueError(f"No video files found in {values['data_dir']}.")

        logger.info(f"Found {len(files)} videos in {values['data_dir']}.")
        values["filepaths"] = pd.DataFrame(files, columns=["filepath"])
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
            will be unfrozen. Defaults to 5.
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

    unfreeze_backbone_at_epoch: Optional[int] = 5
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
            will be stopped. Defaults to 5.
        verbose (bool): Verbosity mode. Defaults to True.
        mode (str, optional): Options are "min" or "max". In "min" mode, training
            will stop when the quantity monitored has stopped decreasing and in
            "max" mode it will stop when the quantity monitored has stopped increasing.
            If None, mode will be inferred from monitor. Defaults to None.
    """

    monitor: MonitorEnum = "val_macro_f1"
    patience: int = 5
    verbose: bool = True
    mode: Optional[str] = None

    @root_validator
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
            columns called 'filepath' (absolute or relative to the data_dir) and
            'label', and optionally columns called 'split' ("train", "val", or "holdout")
            and 'site'. Labels must be specified to train a model.
        data_dir (DirectoryPath): Path to a directory containing training
            videos. Defaults to the working directory.
        model_name (str, optional): Name of the model to use for training. Options are:
            time_distributed, slowfast, european, blank_nonblank. Defaults to time_distributed.
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
        batch_size (int): Batch size to use for training. Defaults to 2.
        auto_lr_find (bool): Use a learning rate finder algorithm when calling
            trainer.tune() to try to find an optimal initial learning rate. Defaults to
            False. The learning rate finder is not guaranteed to find a good learning
            rate; depending on the dataset, it can select a learning rate that leads to
            poor model training. Use with caution.
        backbone_finetune_params (BackboneFinetuneConfig, optional): Set parameters
            to finetune a backbone model to align with the current learning rate.
            Defaults to a BackboneFinetuneConfig(unfreeze_backbone_at_epoch=5,
            backbone_initial_ratio_lr=0.01, multiplier=1, pre_train_bn=False,
            train_bn=False, verbose=True).
        gpus (int): Number of GPUs to train on applied per node.
            Defaults to all of the available GPUs found on the machine.
        num_workers (int): Number of subprocesses to use for data loading. 0 means
            that the data will be loaded in the main process. The maximum value is
           the number of CPUs in the system. Defaults to 3.
        max_epochs (int, optional): Stop training once this number of epochs is
            reached. Disabled by default (None), which stops training at 1000 epochs.
        early_stopping_config (EarlyStoppingConfig, optional): Configuration for
            early stopping, which monitors a metric during training and stops training
            when the metric stops improving. Defaults to EarlyStoppingConfig(monitor='val_macro_f1',
            patience=5, verbose=True, mode='max').
        weight_download_region (str): s3 region to download pretrained weights from.
            Options are "us" (United States), "eu" (Europe), or "asia" (Asia Pacific).
            Defaults to "us".
        split_proportions (dict): Proportions used to divide data into training,
            validation, and holdout sets if a if a "split" column is not included in
            labels. Defaults to "train": 3, "val": 1, "holdout": 1.
        save_dir (Path, optional): Path to a directory where training files
            will be saved. Files include the best model checkpoint (``model_name``.ckpt),
            training configuration (configuration.yaml), Tensorboard logs
            (events.out.tfevents...), test metrics (test_metrics.json), validation
            metrics (val_metrics.json), and model hyperparameters (hparams.yml).
            If None, a folder is created in the working directory. Defaults to None.
        overwrite (bool): If True, will save outputs in `save_dir` overwriting if those
            exist. If False, will create auto-incremented `version_n` folder in `save_dir`
            with model outputs. Defaults to False.
        skip_load_validation (bool). Skip ffprobe check, which verifies that all
            videos can be loaded and skips files that cannot be loaded. Defaults
            to False.
        from_scratch (bool): Instantiate the model with base weights. This means
            starting with ImageNet weights for image-based models (time_distributed,
            european, and blank_nonblank) and Kinetics weights for video-based models
            (slowfast). Defaults to False.
        use_default_model_labels (bool, optional): By default, output the full set of
            default model labels rather than just the species in the labels file. Only
            applies if the provided labels are a subset of the default model labels.
            If set to False, will replace the model head for finetuning and output only
            the species in the provided labels file.
        model_cache_dir (Path, optional): Cache directory where downloaded model weights
            will be saved. If None and the MODEL_CACHE_DIR environment variable is
            not set, uses your default cache directory. Defaults to None.
    """

    labels: Union[FilePath, pd.DataFrame]
    data_dir: DirectoryPath = ""
    checkpoint: Optional[FilePath] = None
    scheduler_config: Optional[Union[str, SchedulerConfig]] = "default"
    model_name: Optional[ModelEnum] = ModelEnum.time_distributed.value
    dry_run: Union[bool, int] = False
    batch_size: int = 2
    auto_lr_find: bool = False
    backbone_finetune_config: Optional[BackboneFinetuneConfig] = BackboneFinetuneConfig()
    gpus: int = GPUS_AVAILABLE
    num_workers: int = 3
    max_epochs: Optional[int] = None
    early_stopping_config: Optional[EarlyStoppingConfig] = EarlyStoppingConfig()
    weight_download_region: RegionEnum = "us"
    split_proportions: Optional[Dict[str, int]] = {"train": 3, "val": 1, "holdout": 1}
    save_dir: Path = Path.cwd()
    overwrite: bool = False
    skip_load_validation: bool = False
    from_scratch: bool = False
    use_default_model_labels: Optional[bool] = None
    model_cache_dir: Optional[Path] = None

    class Config:
        arbitrary_types_allowed = True

    _validate_gpus = validator("gpus", allow_reuse=True, pre=True)(validate_gpus)

    _validate_model_cache_dir = validator("model_cache_dir", allow_reuse=True, always=True)(
        validate_model_cache_dir
    )

    @root_validator(skip_on_failure=True)
    def validate_from_scratch_and_checkpoint(cls, values):
        if values["from_scratch"]:
            if values["checkpoint"] is not None:
                raise ValueError("If from_scratch=True, you cannot specify a checkpoint.")

            if values["model_name"] is None:
                raise ValueError("If from_scratch=True, model_name cannot be None.")

        return values

    _validate_model_name_and_checkpoint = root_validator(allow_reuse=True, skip_on_failure=True)(
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

        # subset to required and optional
        cols_to_keep = [c for c in labels.columns if c in ["filepath", "label", "site", "split"]]
        labels = labels[cols_to_keep]

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
                    "Labels contains split column yet split_proportions are also provided. Split column in labels takes precedence."
                )
                # set to None for clarity in final configuration.yaml
                values["split_proportions"] = None

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
            data_dir=values["data_dir"],
            skip_load_validation=values["skip_load_validation"],
        )
        return values

    @root_validator(skip_on_failure=True)
    def validate_provided_species_and_use_default_model_labels(cls, values):
        """If the model species are the desired output, the labels file must contain
        a subset of the model species.
        """
        provided_species = set(values["labels"].label)
        model_species = set(
            get_model_species(checkpoint=values["checkpoint"], model_name=values["model_name"])
        )

        if not provided_species.issubset(model_species):
            # if labels are not a subset, user cannot set use_default_model_labels to True
            if values["use_default_model_labels"]:
                raise ValueError(
                    "Conflicting information between `use_default_model_labels=True` and the "
                    "species provided in labels file. "
                    "If you want your model to predict all the zamba species, make sure your "
                    "labels are a subset. The species in the labels file that are not "
                    f"in the model species are {provided_species - model_species}. "
                    "If you want your model to only predict the species in your labels file, "
                    "set `use_default_model_labels` to False."
                )

            else:
                values["use_default_model_labels"] = False

        # if labels are a subset, default to True if no value provided
        elif values["use_default_model_labels"] is None:
            values["use_default_model_labels"] = True

        return values

    @root_validator(skip_on_failure=True)
    def preprocess_labels(cls, values):
        """One hot encode, add splits, and check for binary case.

        Replaces values['labels'] with modified DataFrame.

        Args:
            values: dictionary containing 'labels' and other config info
        """
        logger.info("Preprocessing labels into one hot encoded labels with one row per video.")
        labels = values["labels"]

        # lowercase to facilitate subset checking
        labels["label"] = labels.label.str.lower()

        model_species = get_model_species(
            checkpoint=values["checkpoint"], model_name=values["model_name"]
        )
        labels["label"] = pd.Categorical(
            labels.label, categories=model_species if values["use_default_model_labels"] else None
        )
        # one hot encode collapse to one row per video
        labels = (
            pd.get_dummies(labels.rename(columns={"label": "species"}), columns=["species"])
            .groupby("filepath")
            .max()
        )

        # if no "split" column, set up train, val, and holdout split
        if "split" not in labels.columns:
            make_split(labels, values)

        # if there are only two species columns and every video belongs to one of them,
        # keep only blank label if it exists to allow resuming of blank_nonblank model
        # otherwise drop the second species column so the problem is treated as a binary classification
        species_cols = labels.filter(regex="species_").columns
        sums = labels[species_cols].sum(axis=1)

        if len(species_cols) == 2 and (sums == 1).all():
            col_to_keep = "species_blank" if "species_blank" in species_cols else species_cols[0]
            col_to_drop = [c for c in species_cols if c != col_to_keep]

            logger.warning(
                f"Binary case detected so only one species column will be kept. Output will be the binary case of {col_to_keep}."
            )
            labels = labels.drop(columns=col_to_drop)

        # filepath becomes column instead of index
        values["labels"] = labels.reset_index()
        return values


def make_split(labels, values):
    """Add a split column to `labels`.

    Args:
        labels: DataFrame with one row per video
        values: dictionary with config info
    """
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
        # otherwise randomly allocate
        logger.info(
            "No 'site' column found so videos for each species will be randomly allocated across splits using provided split proportions."
        )

        expected_splits = [k for k, v in values["split_proportions"].items() if v > 0]
        random.seed(SPLIT_SEED)

        # check we have at least as many videos per species as we have splits
        # labels are OHE at this point
        num_videos_per_species = labels.filter(regex="species_").sum().to_dict()
        too_few = {
            k.split("species_", 1)[1]: v
            for k, v in num_videos_per_species.items()
            if 0 < v < len(expected_splits)
        }

        if len(too_few) > 0:
            raise ValueError(
                f"Not all species have enough videos to allocate into the following splits: {', '.join(expected_splits)}. A minimum of {len(expected_splits)} videos per label is required. Found the following counts: {too_few}. Either remove these labels or add more videos."
            )

        for c in labels.filter(regex="species_").columns:
            species_df = labels[labels[c] > 0]

            if len(species_df):
                # within each species, seed splits by putting one video in each set and then allocate videos based on split proportions
                labels.loc[species_df.index, "split"] = expected_splits + random.choices(
                    list(values["split_proportions"].keys()),
                    weights=list(values["split_proportions"].values()),
                    k=len(species_df) - len(expected_splits),
                )

        logger.info(f"{labels.split.value_counts()}")

    # write splits.csv
    filename = values["save_dir"] / "splits.csv"
    logger.info(f"Writing out split information to {filename}.")

    # create the directory to save if we need to
    values["save_dir"].mkdir(parents=True, exist_ok=True)

    labels.reset_index()[["filepath", "split"]].drop_duplicates().to_csv(filename, index=False)


class PredictConfig(ZambaBaseModel):
    """
    Configuration for using a model for inference.

    Args:
        filepaths (FilePath): Path to a CSV containing videos for inference, with
            one row per video in the data_dir. There must be a column called
            'filepath' (absolute or relative to the data_dir). If None, uses
            all files in data_dir. Defaults to None.
        data_dir (DirectoryPath): Path to a directory containing videos for
            inference. Defaults to the working directory.
        model_name (str, optional): Name of the model to use for inference. Options are:
            time_distributed, slowfast, european, blank_nonblank. Defaults to time_distributed.
        checkpoint (FilePath, optional): Path to a custom checkpoint file (.ckpt)
            generated by zamba that can be used to generate predictions. If None,
            defaults to a pretrained model. Defaults to None.
        gpus (int): Number of GPUs to use for inference.
            Defaults to all of the available GPUs found on the machine.
        num_workers (int): Number of subprocesses to use for data loading. 0 means
            that the data will be loaded in the main process. The maximum value is
            the number of CPUs in the system. Defaults to 3.
        batch_size (int): Batch size to use for inference. Defaults to 2.
        save (bool): Whether to save out predictions. If False, predictions are
            not saved. Defaults to True.
        save_dir (Path, optional): An optional directory in which to save the model
             predictions and configuration yaml. If no save_dir is specified and save=True,
             outputs will be written to the current working directory. Defaults to None.
        overwrite (bool): If True, overwrite outputs in save_dir if they exist.
            Defaults to False.
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
            Options are "us" (United States), "eu" (Europe), or "asia" (Asia Pacific).
            Defaults to "us".
        skip_load_validation (bool): By default, zamba runs a check to verify that
            all videos can be loaded and skips files that cannot be loaded. This can
            be time intensive, depending on how many videos there are. If you are very
            confident all your videos can be loaded, you can set this to True and skip
            this check. Defaults to False.
        model_cache_dir (Path, optional): Cache directory where downloaded model weights
            will be saved. If None and no environment variable is set, will use your
            default cache directory. Defaults to None.
    """

    data_dir: DirectoryPath = ""
    filepaths: Optional[FilePath] = None
    checkpoint: Optional[FilePath] = None
    model_name: Optional[ModelEnum] = ModelEnum.time_distributed.value
    gpus: int = GPUS_AVAILABLE
    num_workers: int = 3
    batch_size: int = 2
    save: bool = True
    save_dir: Optional[Path] = None
    overwrite: bool = False
    dry_run: bool = False
    proba_threshold: Optional[float] = None
    output_class_names: bool = False
    weight_download_region: RegionEnum = "us"
    skip_load_validation: bool = False
    model_cache_dir: Optional[Path] = None

    _validate_gpus = validator("gpus", allow_reuse=True, pre=True)(validate_gpus)

    _validate_model_cache_dir = validator("model_cache_dir", allow_reuse=True, always=True)(
        validate_model_cache_dir
    )

    @root_validator(skip_on_failure=True)
    def validate_dry_run_and_save(cls, values):
        if values["dry_run"] and (
            (values["save"] is not False) or (values["save_dir"] is not None)
        ):
            logger.warning(
                "Cannot save when predicting with dry_run=True. Setting save=False and save_dir=None."
            )
            values["save"] = False
            values["save_dir"] = None

        return values

    @root_validator(skip_on_failure=True)
    def validate_save_dir(cls, values):
        save_dir = values["save_dir"]
        save = values["save"]

        # if no save_dir but save is True, use current working directory
        if save_dir is None and save:
            save_dir = Path.cwd()

        if save_dir is not None:
            # check if files exist
            if (
                (save_dir / "zamba_predictions.csv").exists()
                or (save_dir / "predict_configuration.yaml").exists()
            ) and not values["overwrite"]:
                raise ValueError(
                    f"zamba_predictions.csv and/or predict_configuration.yaml already exist in {save_dir}. If you would like to overwrite, set overwrite=True"
                )

            # make a directory if needed
            save_dir.mkdir(parents=True, exist_ok=True)

            # set save to True if save_dir is set
            if not save:
                save = True

        values["save_dir"] = save_dir
        values["save"] = save

        return values

    _validate_model_name_and_checkpoint = root_validator(allow_reuse=True, skip_on_failure=True)(
        validate_model_name_and_checkpoint
    )

    @root_validator(skip_on_failure=True)
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

    _get_filepaths = root_validator(allow_reuse=True, pre=False, skip_on_failure=True)(
        get_filepaths
    )

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
        else:
            files_df = files_df[["filepath"]]

        # can only contain one row per filepath
        duplicated = files_df.filepath.duplicated()
        if duplicated.sum() > 0:
            logger.warning(
                f"Found {duplicated.sum():,} duplicate row(s) in filepaths csv. Dropping duplicates so predictions will have one row per video."
            )
            files_df = files_df[["filepath"]].drop_duplicates()

        values["filepaths"] = check_files_exist_and_load(
            df=files_df,
            data_dir=values["data_dir"],
            skip_load_validation=values["skip_load_validation"],
        )
        return values


class ModelConfig(ZambaBaseModel):
    """Contains all configs necessary to use a model for training or inference.
    Must contain a train_config or a predict_config at a minimum.

    Args:
        video_loader_config (VideoLoaderConfig, optional): An instantiated VideoLoaderConfig.
            If None, will use default video loader config for model specified in TrainConfig or
            PredictConfig.
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

    @root_validator(skip_on_failure=True)
    def one_config_must_exist(cls, values):
        if values["train_config"] is None and values["predict_config"] is None:
            raise ValueError("Must provide either `train_config` or `predict_config`.")
        else:
            return values

    @root_validator(skip_on_failure=True)
    def get_default_video_loader_config(cls, values):
        if values["video_loader_config"] is None:
            model_name = (
                values["train_config"].model_name
                if values["train_config"] is not None
                else values["predict_config"].model_name
            )

            logger.info(f"No video loader config specified. Using default for {model_name}.")

            config_file = MODELS_DIRECTORY / f"{model_name}/config.yaml"
            with config_file.open() as f:
                config_dict = yaml.safe_load(f)

            values["video_loader_config"] = VideoLoaderConfig(**config_dict["video_loader_config"])

        return values
