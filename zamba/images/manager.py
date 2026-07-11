from collections.abc import Iterable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from functools import partial
import sys

import git
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger

try:
    from megadetector.detection import run_detector
except ModuleNotFoundError:
    from detection import run_detector
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.tuner.tuning import Tuner
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import ModuleList
from torchvision.transforms import transforms
from tqdm import tqdm
import yaml

from zamba.images.classifier import ImageClassifierModule, infer_model_family
from zamba.images.config import (
    ImageClassificationPredictConfig,
    ImageClassificationTrainingConfig,
    ImageModelEnum,
    ResultsFormat,
)
from zamba.images.data import ImageClassificationDataModule, load_image, absolute_bbox, BboxLayout
from zamba.images.result import results_to_megadetector_format
from zamba.models.instantiation import instantiate_model
from zamba.models.utils import get_checkpoint_hparams
from zamba.pytorch.transforms import resize_and_pad
from zamba.pytorch.utils import configure_inference_determinism


def resolve_inference_family(model_name, checkpoint) -> str:
    """Determine the preprocessing family without trusting a (possibly stale/mangled)
    ``model_name`` default.

    When a checkpoint is provided it is authoritative: the family is read from the
    checkpoint's persisted ``model_family`` / legacy ``zamba_model`` hparam, falling
    back to inference from the stored architecture name. Only when there is no
    checkpoint do we fall back to the configured ``model_name``.
    """
    if checkpoint is not None:
        try:
            hp = get_checkpoint_hparams(checkpoint)
            family = hp.get("model_family") or hp.get("zamba_model")
            if family:
                return family
            return infer_model_family(hp.get("model_name"))
        except Exception as exc:  # noqa: BLE001 -- fall back to model_name on any read error
            logger.warning(f"Could not read family from checkpoint ({exc}); using model_name.")
    return infer_model_family(model_name)


def get_weights(split):
    labels_df = split.filter(like="species_")
    y_array = pd.from_dummies(labels_df).values.flatten()
    classes = labels_df.columns.values
    class_weights = compute_class_weight("balanced", classes=classes, y=y_array)
    return torch.tensor(class_weights).to(torch.float32)


def normalize_prediction_labels(species: list[str]) -> list[str]:
    """Remove the legacy one-hot-encoding prefix from checkpoint labels."""
    return [label.removeprefix("species_") for label in species]


def get_prediction_labels(classifier_module) -> list[str]:
    """Return user-facing labels for new and legacy image checkpoints."""
    if classifier_module.hparams.get("species_labels_are_user_provided", False):
        return classifier_module.species
    return normalize_prediction_labels(classifier_module.species)


def get_default_transforms(model_family: str, image_size=None):
    """Build the (top, bottom) eval transform lists for a preprocessing family.

    The preprocessing is fully determined by ``model_family`` (and the resolved
    ``image_size``), NOT by a model_name string on the config. This lets prediction
    derive transforms directly from a loaded checkpoint. Returns the transform lists
    plus the resolved integer image size.
    """
    # checkpoints may store image_size as a tuple (e.g. (480, 480)); normalize to int
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]

    logger.info(f"Using default transforms for '{model_family}' model family")
    if model_family == ImageModelEnum.SPECIESNET.value:
        # speciesnet is trained on 480x480 images by default
        if image_size is None:
            logger.info("Image size not specified, using value from model checkpoint: 480")
            image_size = 480

        top_transforms = [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ]
        bottom_transforms = [
            transforms.ToTensor(),
        ]
    else:
        # lila.science and generic models: pad to square + ImageNet-ish normalization
        if image_size is None:
            logger.info("Image size not specified, using default value: 224")
            image_size = 224

        top_transforms = [
            transforms.Lambda(partial(resize_and_pad, desired_size=(image_size, image_size))),
        ]
        bottom_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ]

    return top_transforms, bottom_transforms, image_size


def resolve_training_image_size(config: ImageClassificationTrainingConfig):
    """Resolve the image size to train at.

    An explicitly configured ``image_size`` always wins. Otherwise, when finetuning or
    resuming from a checkpoint, the checkpoint's own ``image_size`` takes precedence over
    the preprocessing-family default, since a finetuned model may have been trained at a
    non-default size. Returns ``None`` (deferring to the family default) only when there is
    no explicit size and no usable size on the checkpoint. The returned value may be a
    scalar or a tuple; ``get_default_transforms`` normalizes it to an int.
    """
    if config.image_size is not None:
        return config.image_size

    if config.checkpoint is not None and not config.from_scratch:
        try:
            return get_checkpoint_hparams(config.checkpoint).get("image_size")
        except Exception as exc:  # noqa: BLE001 -- fall back to the family default
            logger.warning(f"Could not read image_size from checkpoint ({exc}); using default.")

    return None


def predict(config: ImageClassificationPredictConfig) -> None:
    configure_inference_determinism(deterministic=config.deterministic)

    logger.info("Loading models")
    detector = run_detector.load_detector("MDV5A", force_cpu=(os.getenv("RUNNER_OS") == "macOS"))
    classifier_module = instantiate_model(
        checkpoint=config.checkpoint,
    )
    classifier_module.eval()
    output_species = get_prediction_labels(classifier_module)

    # The loaded checkpoint is the source of truth for preprocessing. Derive the family
    # (and image size) from the module rather than from config.model_name, which is
    # unreliable when predicting from a bare --checkpoint (it defaults to lila.science).
    # Every real ImageClassifierModule sets a string `model_family`; fall back to the
    # arch name / config only if the loaded object doesn't expose usable values.
    model_family = getattr(classifier_module, "model_family", None)
    if not isinstance(model_family, str):
        base_model_name = getattr(classifier_module, "base_model_name", None)
        model_family = infer_model_family(
            base_model_name if isinstance(base_model_name, str) else config.model_name
        )

    image_size = config.image_size
    if image_size is None:
        module_image_size = getattr(classifier_module, "image_size", None)
        if isinstance(module_image_size, (int, tuple, list)):
            image_size = module_image_size

    top_transforms, bottom_transforms, config.image_size = get_default_transforms(
        model_family, image_size
    )
    image_transforms = transforms.Compose(top_transforms + bottom_transforms)

    logger.info("Running inference")
    predictions = []
    assert isinstance(config.filepaths, pd.DataFrame)

    for filepath in tqdm(sorted(config.filepaths["filepath"].tolist())):
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
        species_df.columns = output_species
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
                df, output_species
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

    model_family = resolve_inference_family(config.model_name, config.checkpoint)
    config.image_size = resolve_training_image_size(config)
    top_transforms, bottom_transforms, config.image_size = get_default_transforms(
        model_family, config.image_size
    )

    if config.extra_train_augmentations:
        augment_transforms = [
            transforms.RandomResizedCrop(
                size=(config.image_size, config.image_size), scale=(0.75, 1.0)
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply(ModuleList([transforms.RandomRotation((-22, 22))]), p=0.2),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomEqualize(p=0.05),
            transforms.RandomAutocontrast(p=0.05),
            transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.05),  # < 1 is more blurry
        ]

        # add random erasing to the end of the pipeline
        final_transforms = [
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]

    # defaults simple transforms are specifically chosen for camera trap imagery
    # - random perspective shift
    # - random horizontal flip (no vertical flip; unlikely animals appear upside down cuz gravity)
    # - random rotation
    else:
        augment_transforms = [
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply(ModuleList([transforms.RandomRotation((-22, 22))]), p=0.2),
        ]
        final_transforms = []

    validation_transforms = transforms.Compose(top_transforms + bottom_transforms)
    train_transforms = transforms.Compose(
        top_transforms + augment_transforms + bottom_transforms + final_transforms
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

    mlflow_logger = False
    try:
        import mlflow
        from pytorch_lightning.loggers import MLFlowLogger

        # Enable system metrics logging in MLflow
        mlflow.enable_system_metrics_logging()
        mlflow_logger = MLFlowLogger(
            run_name=f"zamba-{config.name}-{config.model_name}-{config.lr}-{random.randint(1000, 9999)}",
            experiment_name=config.name,
            tracking_uri=config.mlflow_tracking_uri,
        )
    except Exception as exc:
        logger.warning(
            "MLflow is unavailable; training will continue without MLflow logging. Reason: {}",
            exc,
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
        loss_fn = torch.nn.CrossEntropyLoss(weight=get_weights(data.annotations), reduction="mean")

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
            batch_size=config.batch_size,
        )

    # New checkpoints store the original label names rather than the temporary
    # one-hot column names. This distinguishes them from legacy checkpoints whose
    # labels need output-time normalization.
    classifier_module.hparams["species_labels_are_user_provided"] = True

    # Compile only the inner backbone (not the whole LightningModule) for faster
    # performance; disabled for MacOS and Windows (unsupported/unreliable). Compiling
    # the LightningModule makes torch dynamo trace training_step, which calls
    # self.log() and fails because Lightning introspects the hook with inspect.
    classifier = classifier_module
    if sys.platform not in ("darwin", "win32"):
        try:
            torch._dynamo.config.cache_size_limit = (
                16  # cache more functions than default 8 to avoid recompiling
            )
        except Exception:
            logger.warning("Could not configure torch dynamo cache size limit")
        classifier_module.model = torch.compile(classifier_module.model)

    # lower precision multiplication to speed up training
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        logger.warning("Could not set float32 matmul precision to medium")

    # Set log_every_n_steps to a reasonable value (e.g., 1/10th of batches, minimum of 1)
    log_every_n_steps = max(1, num_training_batches // 10)

    # Use DDP only for genuine multi-device runs. Note that `devices` may be the
    # string "auto" (the default), which is iterable but NOT multi-device, so it must
    # not fall into the Iterable branch below or single-GPU jobs wrongly run under DDP.
    if isinstance(config.devices, int) and config.devices > 1:
        strategy = "ddp"
    elif (
        not isinstance(config.devices, str)
        and isinstance(config.devices, Iterable)
        and len(list(config.devices)) > 1
    ):
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
