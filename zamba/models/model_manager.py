from datetime import datetime
import json
import os
from pathlib import Path
from typing import Optional, Union
import yaml

import git
from loguru import logger
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import torch

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.config import (
    ModelConfig,
    MODEL_MAPPING,
    SchedulerConfig,
    TrainConfig,
    PredictConfig,
    RegionEnum,
)
from zamba_algorithms.models.efficientnet_models import (
    TimeDistributedEfficientNet,
    TimeDistributedEfficientNetMultiLayerHead,
)
from zamba_algorithms.models.i3d_models import I3D
from zamba_algorithms.models.resnet_models import (
    ResnetR2Plus1d18,
    SingleFrameResnet50,
    TimeDistributedResnet50,
)
from zamba_algorithms.models.slowfast_models import SlowFast
from zamba_algorithms.models.x3d_models import X3D
from zamba_algorithms.models.utils import download_weights
from zamba_algorithms.mnist.dataloaders import MNISTDataModule
from zamba_algorithms.mnist.transforms import (
    MNISTOneHot,
    mnist_transforms,
    slowfast_mnist_transforms,
)
from zamba_algorithms.pytorch.finetuning import BackboneFinetuning
from zamba_algorithms.pytorch_lightning.utils import (
    available_models,
    ZambaDataModule,
    ZambaVideoClassificationLightningModule,
)

try:
    from zamba_algorithms.models.timesformer_models import TimeSformer

    TIMESFORMER_AVAILABLE = True
except ImportError:
    TIMESFORMER_AVAILABLE = False
    pass

MODEL_DICT = {
    "i3d": I3D,
    "resnet_r2plus1d": ResnetR2Plus1d18,
    "single_frame_resnet": SingleFrameResnet50,
    "slowfast": SlowFast,
    "time_distributed_efficientnet": TimeDistributedEfficientNet,
    "time_distributed_efficientnet_multilayer_head": TimeDistributedEfficientNetMultiLayerHead,
    "time_distributed_resnet": TimeDistributedResnet50,
    "timesformer": TimeSformer if TIMESFORMER_AVAILABLE else None,
    "x3d": X3D,
}

time_distributed_mnist_transformms = mnist_transforms(
    three_channels=True, repeat=16, time_first=True
)

MNIST_TRANSFORMS = {
    "resnet_2dplus1d": mnist_transforms(three_channels=True, repeat=16, time_first=False),
    "single_frame_resnet": mnist_transforms(),
    "slowfast": slowfast_mnist_transforms(),
    "time_distributed_efficientnet": time_distributed_mnist_transformms,
    "time_distributed_efficientnet_multilayer_head": time_distributed_mnist_transformms,
    "time_distributed_resnet": time_distributed_mnist_transformms,
    "timesformer": mnist_transforms(repeat=16, time_first=False, resize=(224, 224)),
}


def prepare_mnist_module(model_class, **params):
    return MNISTDataModule(
        img_transform=MNIST_TRANSFORMS[model_class], target_transform=MNISTOneHot(), **params
    )


def instantiate_model(
    checkpoint: Union[os.PathLike, str],
    weight_download_region: RegionEnum,
    scheduler_config: Optional[SchedulerConfig],
    cache_dir: Optional[os.PathLike],
    labels: Optional[pd.DataFrame],
    from_scratch: bool = False,
    predict_all_zamba_species: bool = True,
) -> ZambaVideoClassificationLightningModule:
    """Instantiates the model from a checkpoint and detects whether the model head should be replaced.
    The model head is replaced if labels contain species that are not on the model or predict_all_zamba_species=False.

    Supports model instantiation for the following cases:
    - train from scratch (from_scratch=True)
    - finetune with new species (from_scratch=False, labels contains different species than model)
    - finetune with a subset of zamba species and output only the species in the labels file (predict_all_zamba_species=False)
    - finetune with a subset of zamba species but output all zamba species (predict_all_zamba_species=True)
    - predict using pretrained model (labels=None)

    Args:
        checkpoint (path or str): Either the path to a checkpoint on disk or the name of a
            checkpoint file in the S3 bucket, i.e., one that is discoverable by `download_weights`.
        weight_download_region (RegionEnum): Server region for downloading weights.
        scheduler_config (SchedulerConfig, optional): SchedulerConfig to use for training or finetuning.
            Only used if labels is not None.
        cache_dir (path, optional): Directory in which to store pretrained model weights.
        labels (pd.DataFrame, optional): Dataframe where filepath is the index and columns are one hot encoded species.
        from_scratch (bool): Whether to instantiate the model with base weights. This means starting
            from the imagenet weights for image based models and the Kinetics weights for video models.
            Defaults to False. Only used if labels is not None.
        predict_all_zamba_species(bool): Whether the species outputted by the model should be all zamba species.
            If you want the model classes to only be the species in your labels file, set to False.
            Defaults to True. Only used if labels is not None.

    Returns:
        ZambaVideoClassificationLightningModule: Instantiated model
    """
    if not Path(checkpoint).exists():
        logger.info("Downloading weights for model.")
        checkpoint = download_weights(
            filename=str(checkpoint),
            weight_region=weight_download_region,
            destination_dir=cache_dir,
        )

    hparams = torch.load(checkpoint, map_location=torch.device("cpu"))["hyper_parameters"]
    model_class = available_models[hparams["model_class"]]

    logger.info(f"Instantiating model: {model_class.__name__}")

    if labels is None:
        # predict; load from checkpoint uses associated hparams
        logger.info("Loading from checkpoint.")
        return model_class.load_from_checkpoint(checkpoint_path=checkpoint)

    # get species from labels file
    species = labels.filter(regex=r"^species_").columns.tolist()

    # check if species in label file are a subset of pretrained model species
    is_subset = set(species).issubset(set(hparams["species"]))

    # train from scratch
    if from_scratch:
        logger.info("Training from scratch.")

        # default would use scheduler used for pretrained model
        if scheduler_config != "default":
            hparams.update(scheduler_config.dict())

        hparams.update({"species": species})
        model = model_class(**hparams)

    # replace the head
    elif not predict_all_zamba_species or not is_subset:

        if not predict_all_zamba_species:
            logger.info(
                "Limiting only to species in labels file. Replacing model head and finetuning."
            )
        else:
            logger.info(
                "Provided species do not fully overlap with Zamba species. Replacing model head and finetuning."
            )

        # update in case we want to finetune with different scheduler
        if scheduler_config != "default":
            hparams.update(scheduler_config.dict())

        hparams.update({"species": species})
        model = model_class(finetune_from=checkpoint, **hparams)

    # resume training; add additional species columns to labels file if needed
    elif is_subset:
        logger.info(
            "Provided species fully overlap with Zamba species. Resuming training from latest checkpoint."
        )
        # update in case we want to resume with different scheduler
        if scheduler_config != "default":
            hparams.update(scheduler_config.dict())

        model = model_class.load_from_checkpoint(checkpoint_path=checkpoint, **hparams)

        # add in remaining columns for species that are not present
        for c in set(hparams["species"]).difference(set(species)):
            labels[c] = 0

        # sort columns so columns on dataloader are the same as columns on model
        labels.sort_index(axis=1, inplace=True)

    logger.info(f"Using learning rate scheduler: {model.hparams['scheduler']}")
    logger.info(f"Using scheduler params: {model.hparams['scheduler_params']}")

    return model


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
    video_loader_config: VideoLoaderConfig,
):
    """Trains a model.

    Args:
        train_config (TrainConfig): Pydantic config for training.
        video_loader_config (VideoLoaderConfig): Pydantic config for preprocessing videos.
    """

    # set up model
    model = instantiate_model(
        checkpoint=train_config.checkpoint,
        scheduler_config=train_config.scheduler_config,
        weight_download_region=train_config.weight_download_region,
        cache_dir=train_config.cache_dir,
        labels=train_config.labels,
        from_scratch=train_config.from_scratch,
        predict_all_zamba_species=train_config.predict_all_zamba_species,
    )

    data_module = ZambaDataModule(
        video_loader_config=video_loader_config,
        transform=MODEL_MAPPING[train_config.model_name]["transform"],
        train_metadata=train_config.labels,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )

    validate_species(model, data_module)

    train_config.save_directory.mkdir(parents=True, exist_ok=True)

    # add folder version_n that auto increments if we are not overwriting
    tensorboard_version = (
        train_config.save_directory.name if train_config.overwrite_save_directory else None
    )
    tensorboard_save_dir = (
        train_config.save_directory.parent
        if train_config.overwrite_save_directory
        else train_config.save_directory
    )

    tensorboard_logger = TensorBoardLogger(
        save_dir=tensorboard_save_dir,
        name=None,
        version=tensorboard_version,
        default_hp_metric=False,
    )

    logging_and_save_dir = (
        tensorboard_logger.log_dir
        if not train_config.overwrite_save_directory
        else train_config.save_directory
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=logging_and_save_dir,
        filename=train_config.model_name,
        monitor=train_config.early_stopping_config.monitor,
        mode=train_config.early_stopping_config.mode,
    )

    callbacks = [model_checkpoint]

    if train_config.early_stopping_config is not None:
        callbacks.append(EarlyStopping(**train_config.early_stopping_config.dict()))

    if train_config.backbone_finetune_config is not None:
        callbacks.append(BackboneFinetuning(**train_config.backbone_finetune_config.dict()))

    trainer = pl.Trainer(
        gpus=train_config.gpus,
        max_epochs=train_config.max_epochs,
        auto_lr_find=train_config.auto_lr_find,
        logger=tensorboard_logger,
        callbacks=callbacks,
        fast_dev_run=train_config.dry_run,
        accelerator="ddp" if data_module.multiprocessing_context is not None else None,
        plugins=DDPPlugin(find_unused_parameters=False)
        if data_module.multiprocessing_context is not None
        else None,
    )

    if train_config.auto_lr_find:
        logger.info("Finding best learning rate.")
        trainer.tune(model, data_module)

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
            test_metrics = trainer.test(dataloaders=trainer.datamodule.test_dataloader())[0]
            with (Path(logging_and_save_dir) / "test_metrics.json").open("w") as fp:
                json.dump(test_metrics, fp, indent=2)

        if trainer.datamodule.val_dataloader() is not None:
            logger.info("Calculating metrics on validation set.")
            val_metrics = trainer.validate(dataloaders=trainer.datamodule.val_dataloader())[0]
            with (Path(logging_and_save_dir) / "val_metrics.json").open("w") as fp:
                json.dump(val_metrics, fp, indent=2)

    return trainer


def predict_model(
    predict_config: PredictConfig,
    video_loader_config: VideoLoaderConfig,
):
    """Predicts from a model and writes out predictions to a csv.

    Args:
        predict_config (PredictConfig): Pydantic config for performing inference.
        video_loader_config (VideoLoaderConfig): Pydantic config for preprocessing videos.
    """

    # set up model
    model = instantiate_model(
        checkpoint=predict_config.checkpoint,
        weight_download_region=predict_config.weight_download_region,
        cache_dir=predict_config.cache_dir,
        scheduler_config=None,
        labels=None,
    )

    data_module = ZambaDataModule(
        video_loader_config=video_loader_config,
        transform=MODEL_MAPPING[predict_config.model_name]["transform"],
        predict_metadata=predict_config.filepaths,
        batch_size=predict_config.batch_size,
        num_workers=predict_config.num_workers,
    )

    validate_species(model, data_module)

    trainer = pl.Trainer(
        gpus=predict_config.gpus, logger=False, fast_dev_run=predict_config.dry_run
    )

    configuration = {
        "model_class": model.model_class,
        "species": model.species,
        "predict_config": json.loads(predict_config.json(exclude={"filepaths"})),
        "inference_start_time": datetime.utcnow().isoformat(),
        "video_loader_config": json.loads(video_loader_config.json()),
    }

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

        config_path = predict_config.save.parent / "predict_configuration.yaml"
        config_path.parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"Writing out full configuration to {config_path}.")
        with config_path.open("w") as fp:
            yaml.dump(configuration, fp)

        logger.info(f"Saving out predictions to {predict_config.save}.")
        with predict_config.save.open("w") as fp:
            df.to_csv(fp, index=True)

    return df


class ModelManager(object):
    """Mediates loading, configuration, and logic of model calls.

    Args:
        config (ModelConfig) : Instantiated ModelConfig.
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
