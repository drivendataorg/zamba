"""Shared configuration utilities used by both image and video configs.

This module contains config classes and validators that do NOT depend on
video-only packages (ffmpeg, av, opencv, pytorchvideo, yolox).
"""

from enum import Enum
from pathlib import Path
import random
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd
from pandas_path import path  # noqa: F401  -- registers pandas .path accessor
from pydantic import BaseModel, DirectoryPath, validator
from pqdm.threads import pqdm
import torch

from zamba.data.metadata import create_site_specific_splits
from zamba.models.utils import (  # noqa: F401
    download_weights,
    get_checkpoint_hparams,
    get_model_checkpoint_filename,
    get_model_species,  # re-exported for zamba.models.config
    RegionEnum,  # re-exported for zamba.models.config
)
from zamba.settings import IMAGE_SUFFIXES, SPLIT_SEED, get_model_cache_dir

GPUS_AVAILABLE = torch.cuda.device_count()


class ModelEnum(str, Enum):
    """Shorthand names of models supported by zamba."""

    time_distributed = "time_distributed"
    slowfast = "slowfast"
    european = "european"
    blank_nonblank = "blank_nonblank"


class MonitorEnum(str, Enum):
    val_macro_f1 = "val_macro_f1"
    val_loss = "val_loss"


def validate_gpus(gpus: int):
    if gpus > GPUS_AVAILABLE:
        raise ValueError(f"Found only {GPUS_AVAILABLE} GPU(s). Cannot use {gpus}.")
    return gpus


def validate_model_cache_dir(model_cache_dir: Optional[Path]):
    if model_cache_dir is None:
        model_cache_dir = get_model_cache_dir()
    model_cache_dir = Path(model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    return model_cache_dir


def check_files_exist_and_load(
    df: pd.DataFrame, data_dir: DirectoryPath, skip_load_validation: bool
):
    """Check whether files exist. Warn and skip files that don't exist.

    The ffprobe load-check is NOT done here -- it lives in the video-specific
    ``config.py`` which overrides this with an extended version.
    """
    df["filepath"] = str(data_dir) / df.filepath.path

    files_df = df[["filepath"]].drop_duplicates()

    logger.info(f"Checking all {len(files_df):,} filepaths exist. Trying fast file checking...")

    paths = files_df["filepath"].apply(Path)
    exists = pqdm(paths, Path.exists, n_jobs=16)
    exists = np.array(exists)

    if exists.dtype != bool:
        logger.info(
            "Fast file checking failed. Running slower check, which can take 30 seconds per thousand files."
        )
        exists = files_df["filepath"].path.exists()

    invalid_files = files_df[~exists]

    if len(invalid_files) == len(files_df):
        raise ValueError(
            f"None of the video filepaths exist. Are you sure they're specified correctly? "
            f"Here's an example invalid path: {invalid_files.filepath.values[0]}. "
            f"Either specify absolute filepaths in the csv or provide filepaths relative to `data_dir`."
        )
    elif len(invalid_files) > 0:
        logger.debug(
            f"The following files could not be found: {'/n'.join(invalid_files.filepath.values.tolist())}"
        )
        logger.warning(
            f"Skipping {len(invalid_files)} file(s) that could not be found. "
            f"For example, {invalid_files.filepath.values[0]}."
        )
        files_df = files_df[~files_df.filepath.isin(invalid_files.filepath)]

    if not skip_load_validation:
        logger.info(
            "Checking that all media files can be loaded. If you're very confident all "
            "your files can be loaded, you can skip this with `skip_load_validation`, "
            "but it's not recommended."
        )

    df = df[~df.filepath.isin(invalid_files.filepath)].reset_index(drop=True)
    return df


def validate_model_name_and_checkpoint(cls, values):
    """Ensures a checkpoint file or model name is provided."""
    # lazily ensure video models are registered so available_models is populated
    from zamba.models.registry import available_models, ensure_registered

    ensure_registered()

    checkpoint = values.get("checkpoint")
    model_name = values.get("model_name")

    if checkpoint is None and model_name is None:
        raise ValueError("Must provide either model_name or checkpoint path.")

    elif checkpoint is not None and model_name is not None:
        logger.info(f"Using checkpoint file: {checkpoint}.")
        hparams = get_checkpoint_hparams(checkpoint)
        try:
            values["model_name"] = available_models[hparams["model_class"]]._default_model_name
        except (AttributeError, KeyError):
            model_name = f"{model_name}-{checkpoint.stem}"

    elif checkpoint is None and model_name is not None:
        if not values.get("from_scratch"):
            values["checkpoint"] = get_model_checkpoint_filename(model_name)
            cached_path = Path(values["model_cache_dir"]) / values["checkpoint"]
            if cached_path.exists():
                values["checkpoint"] = cached_path

            if not values["checkpoint"].exists():
                logger.info(
                    f"Downloading weights for model '{model_name}' to {values['model_cache_dir']}."
                )
                values["checkpoint"] = download_weights(
                    filename=str(values["checkpoint"]),
                    weight_region=values["weight_download_region"],
                    destination_dir=values["model_cache_dir"],
                )

    return values


def get_filepaths(values, suffix_whitelist):
    """If no file list is passed, get all files in data directory."""
    if values["filepaths"] is None:
        logger.info(f"Getting files in {values['data_dir']}.")
        files = []
        new_suffixes = []

        for f in Path(values["data_dir"]).rglob("*"):
            if f.is_file():
                if f.suffix.lower() in suffix_whitelist:
                    files.append(f.resolve())
                else:
                    new_suffixes.append(f.suffix.lower())

        if len(new_suffixes) > 0:
            logger.warning(
                f"Ignoring {len(new_suffixes)} file(s) with suffixes {set(new_suffixes)}. "
                f"To include, specify all suffixes with a VIDEO_SUFFIXES or IMAGE_SUFFIXES environment variable."
            )

        if len(files) == 0:
            error_msg = f"No relevant files found in {values['data_dir']}."
            if len(set(new_suffixes) & set(IMAGE_SUFFIXES)) > 0:
                error_msg += " Image files *were* found. Use a command starting with `zamba image` to work with images rather than videos."
            raise ValueError(error_msg)

        logger.info(f"Found {len(files):,} media files in {values['data_dir']}.")
        values["filepaths"] = pd.DataFrame(files, columns=["filepath"])
    return values


class ZambaBaseModel(BaseModel):
    """Set defaults for all models that inherit from the pydantic base model."""

    class Config:
        extra = "forbid"
        use_enum_values = True
        validate_assignment = True


class SchedulerConfig(ZambaBaseModel):
    """Configuration for a pytorch learning rate scheduler."""

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
        return scheduler


def make_split(labels, values):
    """Add a split column to `labels`."""
    logger.info(
        f"Dividing media files into train, val, and holdout sets using the following "
        f"split proportions: {values['split_proportions']}."
    )

    if "site" in labels.columns:
        logger.info("Using provided 'site' column to do a site-specific split")
        labels["split"] = create_site_specific_splits(
            labels["site"], proportions=values["split_proportions"]
        )
    else:
        logger.info(
            "No 'site' column found so media files for each species will be randomly "
            "allocated across splits using provided split proportions."
        )

        expected_splits = [k for k, v in values["split_proportions"].items() if v > 0]
        random.seed(SPLIT_SEED)

        num_videos_per_species = labels.filter(regex="species_").sum().to_dict()
        too_few = {
            k.split("species_", 1)[1]: v
            for k, v in num_videos_per_species.items()
            if 0 < v < len(expected_splits)
        }

        if len(too_few) > 0:
            raise ValueError(
                f"Not all species have enough media files to allocate into the following "
                f"splits: {', '.join(expected_splits)}. A minimum of {len(expected_splits)} "
                f"media files per label is required. Found the following counts: {too_few}. "
                f"Either remove these labels or add more images/videos."
            )

        for c in labels.filter(regex="species_").columns:
            species_df = labels[labels[c] > 0]
            if len(species_df):
                labels.loc[species_df.index, "split"] = expected_splits + random.choices(
                    list(values["split_proportions"].keys()),
                    weights=list(values["split_proportions"].values()),
                    k=len(species_df) - len(expected_splits),
                )

        logger.info(f"{labels.split.value_counts()}")

    filename = values["save_dir"] / "splits.csv"
    logger.info(f"Writing out split information to {filename}.")
    values["save_dir"].mkdir(parents=True, exist_ok=True)
    labels.reset_index()[["filepath", "split"]].drop_duplicates().to_csv(filename, index=False)
