from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import git
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from zamba.models.efficientnet_models import (
    TimeDistributedEfficientNet,
    TimeDistributedEfficientNetMultiLayerHead,
)
from zamba.models.slowfast_models import SlowFast
from zamba.tests.mnist.dataloaders import MNISTDataModule
from zamba.tests.mnist.transforms import (
    MNISTOneHot,
    mnist_transforms,
    slowfast_mnist_transforms,
)
from zamba.pytorch.finetuning import BackboneFinetuning
from zamba.settings import MODEL_SCRIPTS


try:
    from zamba.models.timesformer_models import TimeSformer

    TIMESFORMER_AVAILABLE = True
except ImportError:
    TIMESFORMER_AVAILABLE = False
    pass

MODEL_DICT = {
    "slowfast": SlowFast,
    "time_distributed_efficientnet_multilayer_head": TimeDistributedEfficientNetMultiLayerHead,
}

time_distributed_mnist_transformms = mnist_transforms(
    three_channels=True, repeat=16, time_first=True
)

MNIST_TRANSFORMS = {
    "slowfast": slowfast_mnist_transforms(),
    "time_distributed_efficientnet_multilayer_head": time_distributed_mnist_transformms,
}

DEFAULT_BACKBONE_FINETUNE_PARAMS = {
    "unfreeze_backbone_at_epoch": 15,
    "backbone_initial_ratio_lr": 0.01,
    "multiplier": 1,
    "pre_train_bn": False,  # freeze batch norm layers prior to finetuning
    "train_bn": False,  # don't train bn layers in unfrozen finetuning layers
    "verbose": True,
}

DEFAULT_EARLY_STOPPING_PARAMS = {
    "monitor": "val_macro_f1",
    "mode": "max",
    "patience": 3,
    "verbose": True,
}


def prepare_mnist_module(model_class, **params):
    return MNISTDataModule(
        img_transform=MNIST_TRANSFORMS[model_class], target_transform=MNISTOneHot(), **params
    )


def train_model(
    data_module: Union[pl.core.datamodule.LightningDataModule, str],
    model_class: str,
    model_name: str = None,
    model_params: Optional[dict] = None,
    resume_from_checkpoint: Optional[Union[Path, str]] = None,
    backbone_finetune: bool = False,
    backbone_finetune_params: Optional[dict] = None,
    gpus: Optional[Union[List[int], str, int]] = 1,
    max_epochs: int = 20,
    early_stopping: bool = True,
    early_stopping_params: Optional[dict] = None,
    dev_run: Union[bool, int] = False,
    auto_lr_find: bool = True,
    tensorboard_log_dir: str = "tensorboard_logs",
    mnist_datamodule_params: Optional[dict] = None,
):
    """Trains a model.

    Args:
        data_module (LightningDataModule or str): Data module to use for training.
        model_class (str): Name of model to train. (Key to MODEL_DICT.)
        model_name (str, optional): If provided, use as the run name in tensorboard. Otherwise, the
            run will be named using `model_class`.
        model_params (dict, optional): Additional parameters to pass to the model.
        resume_from_checkpoint (Path or str, optional): If provided, resume training from a
            checkpoint. Epoch number is resumed.
        backbone_finetune (bool): If True, finetune the `model.backbone` portion of the model using
            `pytorch_lightning.callbacks.BackboneFinetuning`.
        backbone_finetune_params (dict, optional): Additional parameters for `BackboneFinetuning`.
        gpus (list of int, str, or int, optional)
        max_epochs (int): Maximum number of epochs to train
        early_stopping (bool): If True, implement early stopping.
        early_stopping_params (dict, optional): Additional parameters for `EarlyStopping`.
            Important parameters include:
              - "monitor", the name of metric that training will monitor for early stopping. If
                provided, the model checkpoint callback will also monitor this metric.
              - "mode": Indication of whether early stopping metric should increase ("max") or
                decrease ("min").
        dev_run (bool or int): Runs n if set to n (int) else 1 if set to True batch(es) of train, val and test
            to find any bugs (https://pytorch-lightning.readthedocs.io/en/latest/common/debugging.html#fast-dev-run)
        auto_lr_find (bool): Whether to use PTL's built-in learning rate finder.
        tensorboard_log_dir (str): Name of tensorboard log directory.
        mnist_datamodule_params (dict, optional): Additional parameters for `MNISTDataModule`
    """
    if isinstance(data_module, str):
        if data_module == "mnist":
            mnist_datamodule_params = (
                {} if mnist_datamodule_params is None else mnist_datamodule_params
            )
            data_module = prepare_mnist_module(model_class, **mnist_datamodule_params)
        else:
            raise ValueError(
                "data_module must be `mnist` or an instantiated pl.LightningDataModule."
            )

    model_params = {} if model_params is None else model_params
    model_module = MODEL_DICT[model_class](num_classes=data_module.num_classes, **model_params)

    if model_name is None:
        model_name = model_class

    updated_early_stopping_params = DEFAULT_EARLY_STOPPING_PARAMS.copy()
    updated_early_stopping_params.update(
        {} if early_stopping_params is None else early_stopping_params
    )

    callbacks = [
        ModelCheckpoint(
            monitor=updated_early_stopping_params["monitor"] if early_stopping else "val_loss",
            mode=updated_early_stopping_params["mode"] if early_stopping else "min",
        )
    ]

    if early_stopping:
        callbacks.append(EarlyStopping(**updated_early_stopping_params))

    if backbone_finetune:
        updated_backbone_finetune_params = DEFAULT_BACKBONE_FINETUNE_PARAMS.copy()
        updated_backbone_finetune_params.update(
            {} if backbone_finetune_params is None else backbone_finetune_params
        )
        callbacks.append(BackboneFinetuning(**updated_backbone_finetune_params))

    else:
        updated_backbone_finetune_params = dict()

    logger = TensorBoardLogger(
        MODEL_SCRIPTS / tensorboard_log_dir, name=model_name, default_hp_metric=False
    )

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        auto_lr_find=auto_lr_find,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=dev_run,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    trainer.model_class = model_class
    trainer.model_name = model_name

    if auto_lr_find:
        trainer.tune(model_module, data_module)

    hparams = {
        "auto_lr_find": auto_lr_find,
        "backbone_finetune": backbone_finetune,
        "backbone_finetuning_params": updated_backbone_finetune_params,
        "batch_size": data_module.batch_size,
        "dataset": data_module.dataset_name,
        "early_stopping_params": early_stopping_params,
        "git_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "load_metadata_config": data_module.load_metadata_config,
        "max_epochs": max_epochs,
        "model_class": model_class,
        "model_name": model_name,
        "num_classes": data_module.num_classes,
        "num_workers": data_module.num_workers,
        "predict_metadata": type(data_module.predict_metadata)
        if isinstance(data_module.predict_metadata, pd.DataFrame)
        else data_module.predict_metadata,
        "starting_learning_rate": model_module.lr,
        "training_start_time": datetime.utcnow().isoformat(),
        "train_metadata": type(data_module.train_metadata)
        if isinstance(data_module.train_metadata, pd.DataFrame)
        else data_module.train_metadata,
        "video_loader_config": data_module.video_loader_config,
    }

    # hparams get written out to hparams.yaml file
    # metrics will get logged to TB hparams dashboard as well
    logger.log_hyperparams(
        hparams,
        metrics={
            "val_loss": 0,
            "val_macro_f1": 0,
            "val_top_1_accuracy": 0,
            "val_top_3_accuracy": 0,
            "val_top_5_accuracy": 0,
            "val_top_10_accuracy": 0,
        },
    )

    trainer.fit(model_module, data_module)
    return trainer


def predict_model(
    data_module,
    trainer=None,
    checkpoint: Optional[Path] = None,
    model_class: Optional[str] = None,
    model_name: str = None,
    model_params: Optional[dict] = None,
    gpus: int = 1,
    save: bool = True,
    dev_run: bool = False,
    mnist_datamodule_params: Optional[dict] = None,
):
    dev_run = trainer.fast_dev_run if trainer is not None else dev_run

    if isinstance(data_module, str):
        if data_module == "mnist":
            mnist_datamodule_params = (
                {} if mnist_datamodule_params is None else mnist_datamodule_params
            )
            data_module = prepare_mnist_module(trainer.model_class, **mnist_datamodule_params)
        else:
            raise ValueError(
                "data_module must be `mnist` or an instantiated pl.LightningDataModule."
            )

    if trainer is not None:
        save_dir = Path(trainer.log_dir)
        probas = trainer.predict(dataloaders=data_module.test_dataloader())

    elif checkpoint is not None:
        if (model_class is None) or (model_name is None):
            raise ValueError("You must provide model_class and model_name along with checkpoint.")
        model_params = {} if model_params is None else model_params
        model = MODEL_DICT[model_class]
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint, num_classes=data_module.num_classes, **model_params
        )

        trainer = pl.Trainer(gpus=gpus, logger=False, fast_dev_run=dev_run)
        trainer.model_name = model_name
        # save in same place as checkpoint
        save_dir = Path(checkpoint).parents[1]
        probas = trainer.predict(model=model, dataloaders=data_module.test_dataloader())

    else:
        raise ValueError(
            "You must provide either a checkpoint or a pl.Trainer with trained model."
        )

    if save:
        df = pd.DataFrame(np.vstack(probas))

        test_dataloader = data_module.test_dataloader()

        # don't set columns and index for MNIST test module
        if not isinstance(data_module, MNISTDataModule):
            df.columns = test_dataloader.dataset.targets.columns

            # if we're doing a dev run, don't try to set indices as there will be a length mismatch
            if not dev_run:
                df.index = test_dataloader.dataset.original_indices

        save_path = (
            Path(save_dir)
            / f"{trainer.model_name}_{datetime.utcnow().date().isoformat()}_preds.csv"
        )

        with save_path.open("w+") as f:
            df.to_csv(f, index=True)
