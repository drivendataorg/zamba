from datetime import datetime
import json
import os
from pathlib import Path
from typing import Optional
import yaml

import git
from loguru import logger
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.tuner import Tuner

from zamba.data.video import VideoLoaderConfig
from zamba.models.config import (
    ModelConfig,
    ModelEnum,
    MODEL_MAPPING,
    SchedulerConfig,
    TrainConfig,
    PredictConfig,
)
from zamba.models.registry import available_models
from zamba.models.utils import (
    configure_accelerator_and_devices_from_gpus,
    get_checkpoint_hparams,
    get_default_hparams,
)
from zamba.pytorch.finetuning import BackboneFinetuning
from zamba.pytorch_lightning.utils import ZambaDataModule, ZambaVideoClassificationLightningModule


def instantiate_model(
    checkpoint: os.PathLike,
    labels: Optional[pd.DataFrame] = None,
    scheduler_config: Optional[SchedulerConfig] = None,
    from_scratch: Optional[bool] = None,
    model_name: Optional[ModelEnum] = None,
    use_default_model_labels: Optional[bool] = None,
) -> ZambaVideoClassificationLightningModule:
    """Instantiates the model from a checkpoint and detects whether the model head should be replaced.
    The model head is replaced if labels contain species that are not on the model or use_default_model_labels=False.

    Supports model instantiation for the following cases:
    - train from scratch (from_scratch=True)
    - finetune with new species (from_scratch=False, labels contains different species than model)
    - finetune with a subset of zamba species and output only the species in the labels file (use_default_model_labels=False)
    - finetune with a subset of zamba species but output all zamba species (use_default_model_labels=True)
    - predict using pretrained model (labels=None)

    Args:
        checkpoint (path): Path to a checkpoint on disk.
        labels (pd.DataFrame, optional): Dataframe where filepath is the index and columns are one hot encoded species.
        scheduler_config (SchedulerConfig, optional): SchedulerConfig to use for training or finetuning.
            Only used if labels is not None.
        from_scratch (bool, optional): Whether to instantiate the model with base weights. This means starting
            from the imagenet weights for image based models and the Kinetics weights for video models.
           Only used if labels is not None.
        model_name (ModelEnum, optional): Model name used to look up default hparams used for that model.
            Only relevant if training from scratch.
        use_default_model_labels (bool, optional): Whether to output the full set of default model labels rather than
            just the species in the labels file. Only used if labels is not None.

    Returns:
        ZambaVideoClassificationLightningModule: Instantiated model
    """
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
    species = labels.filter(regex=r"^species_").columns.tolist()
    species = [s.split("species_", 1)[1] for s in species]

    # train from scratch
    if from_scratch:
        logger.info("Training from scratch.")

        # default would use scheduler used for pretrained model
        if scheduler_config != "default":
            hparams.update(scheduler_config.dict())

        hparams.update({"species": species})
        model = model_class(**hparams)
        log_schedulers(model)
        return model

    # determine if finetuning or resuming training

    # check if species in label file are a subset of pretrained model species
    is_subset = set(species).issubset(set(hparams["species"]))

    if is_subset:
        if use_default_model_labels:
            return resume_training(
                scheduler_config=scheduler_config,
                hparams=hparams,
                model_class=model_class,
                checkpoint=checkpoint,
            )

        else:
            logger.info(
                "Limiting only to species in labels file. Replacing model head and finetuning."
            )
            return replace_head(
                scheduler_config=scheduler_config,
                hparams=hparams,
                species=species,
                model_class=model_class,
                checkpoint=checkpoint,
            )

    # without a subset, you will always get a new head
    # the config validation prohibits setting use_default_model_labels to True without a subset
    else:
        logger.info(
            "Provided species do not fully overlap with Zamba species. Replacing model head and finetuning."
        )
        return replace_head(
            scheduler_config=scheduler_config,
            hparams=hparams,
            species=species,
            model_class=model_class,
            checkpoint=checkpoint,
        )


def replace_head(scheduler_config, hparams, species, model_class, checkpoint):
    # update in case we want to finetune with different scheduler
    if scheduler_config != "default":
        hparams.update(scheduler_config.dict())

    hparams.update({"species": species})
    model = model_class(finetune_from=checkpoint, **hparams)
    log_schedulers(model)
    return model


def resume_training(
    scheduler_config,
    hparams,
    model_class,
    checkpoint,
):
    # resume training; add additional species columns to labels file if needed
    logger.info(
        "Provided species fully overlap with Zamba species. Resuming training from latest checkpoint."
    )
    # update in case we want to resume with different scheduler
    if scheduler_config != "default":
        hparams.update(scheduler_config.dict())

    model = model_class.from_disk(path=checkpoint, **hparams)
    log_schedulers(model)
    return model


def log_schedulers(model):
    logger.info(f"Using learning rate scheduler: {model.hparams['scheduler']}")
    logger.info(f"Using scheduler params: {model.hparams['scheduler_params']}")


def validate_species(model: ZambaVideoClassificationLightningModule, data_module: ZambaDataModule):
    conflicts = []
    for dataloader_name, dataloader in zip(
        ("Train", "Val", "Test"),
        (
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            data_module.test_dataloader(),
        ),
    ):
        if (dataloader is not None) and (dataloader.dataset.species != model.species):
            conflicts.append(
                f"""{dataloader_name} dataset includes:\n{", ".join(dataloader.dataset.species)}\n"""
            )

    if len(conflicts) > 0:
        conflicts.append(f"""Model predicts:\n{", ".join(model.species)}""")

        conflict_msg = "\n\n".join(conflicts)
        raise ValueError(
            f"""Dataloader species and model species do not match.\n\n{conflict_msg}"""
        )


def train_model(
    train_config: TrainConfig,
    video_loader_config: Optional[VideoLoaderConfig] = None,
):
    """Trains a model.

    Args:
        train_config (TrainConfig): Pydantic config for training.
        video_loader_config (VideoLoaderConfig, optional): Pydantic config for preprocessing videos.
            If None, will use default for model specified in TrainConfig.
    """
    # get default VLC for model if not specified
    if video_loader_config is None:
        video_loader_config = ModelConfig(
            train_config=train_config, video_loader_config=video_loader_config
        ).video_loader_config

    # set up model
    model = instantiate_model(
        checkpoint=train_config.checkpoint,
        labels=train_config.labels,
        scheduler_config=train_config.scheduler_config,
        from_scratch=train_config.from_scratch,
        model_name=train_config.model_name,
        use_default_model_labels=train_config.use_default_model_labels,
    )

    data_module = ZambaDataModule(
        video_loader_config=video_loader_config,
        transform=MODEL_MAPPING[model.__class__.__name__]["transform"],
        train_metadata=train_config.labels,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )

    validate_species(model, data_module)

    train_config.save_dir.mkdir(parents=True, exist_ok=True)

    # add folder version_n that auto increments if we are not overwriting
    tensorboard_version = train_config.save_dir.name if train_config.overwrite else None
    tensorboard_save_dir = (
        train_config.save_dir.parent if train_config.overwrite else train_config.save_dir
    )

    tensorboard_logger = TensorBoardLogger(
        save_dir=tensorboard_save_dir,
        name=None,
        version=tensorboard_version,
        default_hp_metric=False,
    )

    logging_and_save_dir = (
        tensorboard_logger.log_dir if not train_config.overwrite else train_config.save_dir
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=logging_and_save_dir,
        filename=train_config.model_name,
        monitor=(
            train_config.early_stopping_config.monitor
            if train_config.early_stopping_config is not None
            else None
        ),
        mode=(
            train_config.early_stopping_config.mode
            if train_config.early_stopping_config is not None
            else "min"
        ),
    )

    callbacks = [model_checkpoint]

    if train_config.early_stopping_config is not None:
        callbacks.append(EarlyStopping(**train_config.early_stopping_config.dict()))

    if train_config.backbone_finetune_config is not None:
        callbacks.append(BackboneFinetuning(**train_config.backbone_finetune_config.dict()))

    accelerator, devices = configure_accelerator_and_devices_from_gpus(train_config.gpus)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=train_config.max_epochs,
        logger=tensorboard_logger,
        callbacks=callbacks,
        fast_dev_run=train_config.dry_run,
        strategy=(
            DDPStrategy(find_unused_parameters=False)
            if (data_module.multiprocessing_context is not None) and (train_config.gpus > 1)
            else "auto"
        ),
    )

    if video_loader_config.cache_dir is None:
        logger.info("No cache dir is specified. Videos will not be cached.")
    else:
        logger.info(f"Videos will be cached to {video_loader_config.cache_dir}.")

    if train_config.auto_lr_find:
        logger.info("Finding best learning rate.")
        tuner = Tuner(trainer)
        tuner.lr_find(model=model, datamodule=data_module)

    try:
        git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        git_hash = None

    configuration = {
        "git_hash": git_hash,
        "model_class": model.model_class,
        "species": model.species,
        "starting_learning_rate": model.lr,
        "train_config": json.loads(train_config.json(exclude={"labels"})),
        "training_start_time": datetime.utcnow().isoformat(),
        "video_loader_config": json.loads(video_loader_config.json()),
    }

    if not train_config.dry_run:
        config_path = Path(logging_and_save_dir) / "train_configuration.yaml"
        config_path.parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"Writing out full configuration to {config_path}.")
        with config_path.open("w") as fp:
            yaml.dump(configuration, fp)

    logger.info("Starting training...")
    trainer.fit(model, data_module)

    if not train_config.dry_run:
        if trainer.datamodule.test_dataloader() is not None:
            logger.info("Calculating metrics on holdout set.")
            test_metrics = trainer.test(
                dataloaders=trainer.datamodule.test_dataloader(), ckpt_path="best"
            )[0]
            with (Path(logging_and_save_dir) / "test_metrics.json").open("w") as fp:
                json.dump(test_metrics, fp, indent=2)

        if trainer.datamodule.val_dataloader() is not None:
            logger.info("Calculating metrics on validation set.")
            val_metrics = trainer.validate(
                dataloaders=trainer.datamodule.val_dataloader(), ckpt_path="best"
            )[0]
            with (Path(logging_and_save_dir) / "val_metrics.json").open("w") as fp:
                json.dump(val_metrics, fp, indent=2)

    return trainer


def predict_model(
    predict_config: PredictConfig,
    video_loader_config: VideoLoaderConfig = None,
):
    """Predicts from a model and writes out predictions to a csv.

    Args:
        predict_config (PredictConfig): Pydantic config for performing inference.
        video_loader_config (VideoLoaderConfig, optional): Pydantic config for preprocessing videos.
            If None, will use default for model specified in PredictConfig.
    """
    # get default VLC for model if not specified
    if video_loader_config is None:
        video_loader_config = ModelConfig(
            predict_config=predict_config, video_loader_config=video_loader_config
        ).video_loader_config

    # set up model
    model = instantiate_model(
        checkpoint=predict_config.checkpoint,
    )

    data_module = ZambaDataModule(
        video_loader_config=video_loader_config,
        transform=MODEL_MAPPING[model.__class__.__name__]["transform"],
        predict_metadata=predict_config.filepaths,
        batch_size=predict_config.batch_size,
        num_workers=predict_config.num_workers,
    )

    validate_species(model, data_module)

    if video_loader_config.cache_dir is None:
        logger.info("No cache dir is specified. Videos will not be cached.")
    else:
        logger.info(f"Videos will be cached to {video_loader_config.cache_dir}.")

    accelerator, devices = configure_accelerator_and_devices_from_gpus(predict_config.gpus)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        fast_dev_run=predict_config.dry_run,
    )

    configuration = {
        "model_class": model.model_class,
        "species": model.species,
        "predict_config": json.loads(predict_config.json(exclude={"filepaths"})),
        "inference_start_time": datetime.utcnow().isoformat(),
        "video_loader_config": json.loads(video_loader_config.json()),
    }

    if predict_config.save is not False:
        config_path = predict_config.save_dir / "predict_configuration.yaml"
        logger.info(f"Writing out full configuration to {config_path}.")
        with config_path.open("w") as fp:
            yaml.dump(configuration, fp)

    dataloader = data_module.predict_dataloader()
    logger.info("Starting prediction...")
    probas = trainer.predict(model=model, dataloaders=dataloader)

    df = pd.DataFrame(
        np.vstack(probas), columns=model.species, index=dataloader.dataset.original_indices
    )

    # change output format if specified
    if predict_config.proba_threshold is not None:
        df = (df > predict_config.proba_threshold).astype(int)

    elif predict_config.output_class_names:
        df = df.idxmax(axis=1)

    else:  # round to a useful number of places
        df = df.round(5)

    if predict_config.save is not False:
        preds_path = predict_config.save_dir / "zamba_predictions.csv"
        logger.info(f"Saving out predictions to {preds_path}.")
        with preds_path.open("w") as fp:
            df.to_csv(fp, index=True)

    return df


class ModelManager(object):
    """Mediates loading, configuration, and logic of model calls.

    Args:
        config (ModelConfig): Instantiated ModelConfig.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, config):
        if not isinstance(config, ModelConfig):
            config = ModelConfig.parse_file(config)
        return cls(config)

    def train(self):
        train_model(
            train_config=self.config.train_config,
            video_loader_config=self.config.video_loader_config,
        )

    def predict(self):
        predict_model(
            predict_config=self.config.predict_config,
            video_loader_config=self.config.video_loader_config,
        )
