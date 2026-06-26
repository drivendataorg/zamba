"""Model instantiation shared by both image and video code paths.

This module provides ``instantiate_model`` without importing video-specific
modules (ZambaVideoDataModule, VideoLoaderConfig, etc.), so the image stack
can use it without pulling in the video dependency tree.
"""

import os
from typing import Optional

from loguru import logger
import pandas as pd

from zamba.models.config_common import ModelEnum, SchedulerConfig
from zamba.models.registry import available_models, ensure_registered
from zamba.models.utils import get_checkpoint_hparams, get_default_hparams


def _apply_scheduler_config(hparams, scheduler_config):
    if scheduler_config not in (None, "default"):
        hparams.update(scheduler_config.dict())


def instantiate_model(
    checkpoint: os.PathLike,
    labels: Optional[pd.DataFrame] = None,
    scheduler_config: Optional[SchedulerConfig] = None,
    from_scratch: Optional[bool] = None,
    model_name: Optional[ModelEnum] = None,
    use_default_model_labels: Optional[bool] = None,
    species: Optional[list] = None,
):
    """Instantiate a model from a checkpoint.

    Supports prediction, training from scratch, finetuning, and resume.
    """
    ensure_registered()

    if from_scratch:
        hparams = get_default_hparams(model_name)
    else:
        hparams = get_checkpoint_hparams(checkpoint)

    model_class = available_models[hparams["model_class"]]
    logger.info(f"Instantiating model: {model_class.__name__}")

    # predicting
    if labels is None:
        logger.info("Loading from checkpoint.")
        model = model_class.from_disk(path=checkpoint, **hparams)
        return model

    # get species from labels file
    if species is None:
        species = labels.filter(regex=r"^species_").columns.tolist()
        species = [s.split("species_", 1)[1] for s in species]

    # train from scratch
    if from_scratch:
        logger.info("Training from scratch.")
        _apply_scheduler_config(hparams, scheduler_config)
        hparams.update({"species": species})
        model = model_class(**hparams)
        _log_schedulers(model)
        return model

    # finetuning / resume
    is_subset = set(species).issubset(set(hparams["species"]))

    if is_subset:
        if use_default_model_labels:
            return _resume_training(
                scheduler_config=scheduler_config,
                hparams=hparams,
                model_class=model_class,
                checkpoint=checkpoint,
            )
        else:
            logger.info(
                "Limiting only to species in labels file. Replacing model head and finetuning."
            )
            return _replace_head(
                scheduler_config=scheduler_config,
                hparams=hparams,
                species=species,
                model_class=model_class,
                checkpoint=checkpoint,
            )
    else:
        logger.info(
            "Provided species do not fully overlap with Zamba species. "
            "Replacing model head and finetuning."
        )
        return _replace_head(
            scheduler_config=scheduler_config,
            hparams=hparams,
            species=species,
            model_class=model_class,
            checkpoint=checkpoint,
        )


def _replace_head(scheduler_config, hparams, species, model_class, checkpoint):
    _apply_scheduler_config(hparams, scheduler_config)
    hparams.update({"species": species})
    model = model_class(finetune_from=checkpoint, **hparams)
    _log_schedulers(model)
    return model


def _resume_training(scheduler_config, hparams, model_class, checkpoint):
    logger.info(
        "Provided species fully overlap with Zamba species. "
        "Resuming training from latest checkpoint."
    )
    _apply_scheduler_config(hparams, scheduler_config)
    model = model_class.from_disk(path=checkpoint, **hparams)
    _log_schedulers(model)
    return model


def _log_schedulers(model):
    logger.info(f"Using learning rate scheduler: {model.hparams['scheduler']}")
    logger.info(f"Using scheduler params: {model.hparams['scheduler_params']}")
