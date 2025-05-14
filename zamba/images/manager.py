from collections.abc import Iterable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from functools import partial
import sys

import git
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from megadetector.detection import run_detector
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.tuner.tuning import Tuner
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import ModuleList
from torchvision.transforms import transforms
from tqdm import tqdm
import yaml

from zamba.images.classifier import ImageClassifierModule
from zamba.images.config import (
    ImageClassificationPredictConfig,
    ImageClassificationTrainingConfig,
    ResultsFormat,
)
from zamba.images.data import ImageClassificationDataModule, load_image, absolute_bbox, BboxLayout
from zamba.images.result import results_to_megadetector_format
from zamba.models.model_manager import instantiate_model
from zamba.pytorch.transforms import resize_and_pad


def get_weights(split, all_labels):
    class_weights = compute_class_weight("balanced", classes=all_labels, y=split.label)
    return torch.tensor(class_weights).to(torch.float32)


def predict(config: ImageClassificationPredictConfig) -> None:
    image_transforms = transforms.Compose(
        [
            transforms.Lambda(partial(resize_and_pad, desired_size=config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ]
    )
    logger.info("Loading models")
    detector = run_detector.load_detector("MDV5A", force_cpu=(os.getenv("RUNNER_OS") == "macOS"))
    classifier_module = instantiate_model(
        checkpoint=config.checkpoint,
    )

    logger.info("Running inference")
    predictions = []
    assert isinstance(config.filepaths, pd.DataFrame)

    for filepath in tqdm(config.filepaths["filepath"]):
        image = load_image(filepath)
        results = detector.generate_detections_one_image(
            image, filepath, detection_threshold=config.detections_threshold
        )

        for detection in results["detections"]:
            try:
                bbox = absolute_bbox(image, detection["bbox"], bbox_layout=BboxLayout.XYWH)
                detection_category = detection["category"]
                detection_conf = detection["conf"]
                img = image.crop(bbox)
                input_data = image_transforms(img)

                with torch.no_grad():
                    y_hat = (
                        torch.softmax(classifier_module(input_data.unsqueeze(0)), dim=1)
                        .squeeze(0)
                        .numpy()
                    )
                    predictions.append((filepath, detection_category, detection_conf, bbox, y_hat))
            except Exception as e:
                logger.exception(e)
                continue

    if config.save:
        df = pd.DataFrame(
            predictions,
            columns=["filepath", "detection_category", "detection_conf", "bbox", "result"],
        )
        # Split bbox into separate columns x1, y1, x2, y2
        df[["x1", "y1", "x2", "y2"]] = pd.DataFrame(df["bbox"].tolist(), index=df.index)

        # Split result into separate columns for each class using "species" from classifier module
        species_df = pd.DataFrame(df["result"].tolist(), index=df.index)
        species_df.columns = classifier_module.species
        df = pd.concat([df, species_df], axis=1)

        # Drop the original 'bbox' and 'result' columns
        df = df.drop(columns=["bbox", "result"])

        save_path = config.save_dir / config.results_file_name
        logger.info("Saving results")
        if config.results_file_format == ResultsFormat.CSV:
            save_path = save_path.with_suffix(".csv")
            df.to_csv(save_path, index=False)
        elif config.results_file_format == ResultsFormat.MEGADETECTOR:
            megadetector_format_results = results_to_megadetector_format(
                df, classifier_module.species
            )
            save_path = save_path.with_suffix(".json")
            with open(save_path, "w") as f:
                json.dump(megadetector_format_results.dict(), f)
        logger.info(f"Results saved to {save_path}")


def _save_metrics(
    data: pl.LightningDataModule, trainer: pl.Trainer, model: pl.LightningModule, save_dir: Path
):
    if data.test_dataloader() is not None and len(data.test_dataloader()) > 0:
        logger.info("Calculating metrics on holdout set.")
        test_metrics = trainer.test(model, dataloaders=data.test_dataloader())
        with (save_dir / "test_metrics.json").open("w") as fp:
            json.dump(test_metrics[0], fp, indent=2)

    if data.val_dataloader() is not None and len(data.val_dataloader()) > 0:
        logger.info("Calculating metrics on validation set.")
        val_metrics = trainer.validate(model, dataloaders=data.val_dataloader())
        with (save_dir / "val_metrics.json").open("w") as fp:
            json.dump(val_metrics[0], fp, indent=2)


def _save_config(model, config):
    try:
        git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        git_hash = None

    configuration = {
        "git_hash": git_hash,
        "model_class": model.model_class,
        "species": model.species,
        "starting_learning_rate": model.hparams.lr,
        "train_config": json.loads(config.json(exclude={"labels"})),
        "training_start_time": datetime.now(timezone.utc).isoformat(),
    }

    config_path = config.save_dir / "train_configuration.yaml"
    config_path.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Writing out full configuration to {config_path}.")
    with config_path.open("w") as fp:
        yaml.dump(configuration, fp)


def train(config: ImageClassificationTrainingConfig) -> pl.Trainer:
    if config.save_dir:
        logger.add(
            str(config.save_dir / "training.log"),
            level="INFO",
            format="{time} - {name} - {level} - {message}",
        )

    if config.extra_train_augmentations:
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(partial(resize_and_pad, desired_size=config.image_size)),
                transforms.RandomResizedCrop(
                    size=(config.image_size, config.image_size), scale=(0.75, 1.0)
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomApply(ModuleList([transforms.RandomRotation((-22, 22))]), p=0.2),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomEqualize(p=0.05),
                transforms.RandomAutocontrast(p=0.05),
                transforms.RandomAdjustSharpness(
                    sharpness_factor=0.9, p=0.05
                ),  # < 1 is more blurry
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ]
        )

    # defaults simple transforms are specifically chosen for camera trap imagery
    # - random perspective shift
    # - random horizontal flip (no vertical flip; unlikely animals appear upside down cuz gravity)
    # - random rotation
    else:
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(partial(resize_and_pad, desired_size=config.image_size)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomApply(ModuleList([transforms.RandomRotation((-22, 22))]), p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ]
        )

    validation_transforms = transforms.Compose(
        [
            transforms.Lambda(partial(resize_and_pad, desired_size=config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ]
    )

    os.makedirs(config.checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.checkpoint_path,
        filename=f"zamba-{config.name}-{config.model_name}" + "{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config.early_stopping_patience, mode="min"
    )
    swa = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks = [swa, early_stopping, checkpoint_callback]

    # Enable system metrics logging in MLflow
    mlflow.enable_system_metrics_logging()

    mlflow_logger = MLFlowLogger(
        run_name=f"zamba-{config.name}-{config.model_name}-{config.lr}-{random.randint(1000, 9999)}",
        experiment_name=config.name,
        tracking_uri=config.mlflow_tracking_uri,
    )

    data = ImageClassificationDataModule(
        data_dir=config.data_dir,
        cache_dir=config.cache_dir,
        annotations=config.labels,
        train_transforms=train_transforms,
        test_transforms=validation_transforms,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        detection_threshold=config.detections_threshold,
        crop_images=config.crop_images,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    if config.weighted_loss is True:
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=get_weights(data.annotations, np.unique(config.labels.label)), reduction="mean"
        )

    # Calculate number of training batches
    num_training_batches = len(data.train_dataloader())

    if config.from_scratch:
        initial_lr = config.lr if config.lr is not None else 1e-5  # reasonable empirical default
        classifier_module = ImageClassifierModule(
            model_name=config.model_name,
            species=config.species_in_label_order,
            lr=initial_lr,
            image_size=config.image_size,
            batch_size=config.batch_size if config.batch_size is not None else 16,
            num_training_batches=num_training_batches,
            loss=loss_fn,
            pin_memory=config.accelerator == "gpu",
            scheduler="CosineAnnealingLR",
            scheduler_params={"T_max": config.early_stopping_patience},
        )
    else:
        classifier_module = instantiate_model(
            checkpoint=config.checkpoint,
            labels=config.labels,
            scheduler_config=config.scheduler_config,
            from_scratch=config.from_scratch,
            model_name=None,
            use_default_model_labels=config.use_default_model_labels,
            species=config.species_in_label_order,
        )

    # compile for faster performance; disabled for MacOS which is not supported
    if sys.platform != "darwin":
        classifier = torch.compile(classifier_module)
    else:
        classifier = classifier_module

    # lower precision multiplication to speed up training
    try:
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.cache_size_limit = (
            16  # cache more functions than default 8 to avoid recompiling
        )
    except Exception:
        logger.warning("Could not set float32 matmul precision to medium")

    # Set log_every_n_steps to a reasonable value (e.g., 1/10th of batches, minimum of 1)
    log_every_n_steps = max(1, num_training_batches // 10)

    # get the strategy based on devices
    if isinstance(config.devices, int) and config.devices > 1:
        strategy = "ddp"
    elif isinstance(config.devices, Iterable) and len(config.devices) > 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    if config.accumulated_batch_size is not None:
        accumulate_n_batches = config.accumulated_batch_size // classifier_module.batch_size
    else:
        accumulate_n_batches = 1

    # Create trainers with different configurations
    train_trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=mlflow_logger,
        callbacks=callbacks,
        devices=config.devices,
        accelerator=config.accelerator,
        strategy=strategy,
        log_every_n_steps=log_every_n_steps,
        accumulate_grad_batches=accumulate_n_batches,
    )

    # Single device trainers for lr finding and testing
    single_device_trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=mlflow_logger,
        devices=1,
        accelerator=config.accelerator,
        log_every_n_steps=log_every_n_steps,
        accumulate_grad_batches=accumulate_n_batches,
    )

    tuner = Tuner(single_device_trainer)  # Use single device trainer for tuning

    # find largest feasible batch size if not set explicitly
    if config.batch_size is None:
        # Lightning asserts a batch size deep in its guts, so this makes sure that succeeds
        data.batch_size = 8
        logger.info("Finding maximum batch size")
        tuner.scale_batch_size(
            classifier, datamodule=data, mode="power", init_val=8, steps_per_trial=3
        )
        logger.info(f"Changing batch size to {data.batch_size}")
        # I think only the model gets saved, but the data loader holds the batch size,
        # so we need to make sure these stay in sync.
        classifier.hparams.batch_size = data.batch_size

    # find an optimal learning rate on single device
    if config.lr is None:
        logger.info("Finding a good learning rate")
        lr_finder = tuner.lr_find(classifier, data)
        new_lr = lr_finder.suggestion()
        logger.info(f"Changing learning rate to {new_lr}")
        # Make sure the new learning rate gets saved out as an hparam
        classifier.hparams.lr = new_lr

    _save_config(classifier, config)

    # Train with distributed training
    train_trainer.fit(
        classifier,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    _save_metrics(data, single_device_trainer, classifier, config.save_dir)

    model_path = config.save_dir / (config.model_name + ".ckpt")
    Path(checkpoint_callback.best_model_path).rename(model_path)
    logger.info(f"Model checkpoint saved to {model_path}")


class ZambaImagesManager:
    def predict(self, config: ImageClassificationPredictConfig):
        predict(config)

    def train(self, config: ImageClassificationTrainingConfig):
        train(config)
