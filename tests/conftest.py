import logging
import os
import random
from typing import Optional, Union

from loguru import logger
import pandas as pd
from pathlib import Path
import pytest
from _pytest.logging import caplog as _caplog  # noqa: F401
import torch

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.config import PredictConfig, TrainConfig
from zamba_algorithms.models.megadetector_lite_yolox import MegadetectorLiteYoloX
from zamba_algorithms.models.model_manager import MODEL_MAPPING, train_model
from zamba_algorithms.pytorch.transforms import zamba_image_model_transforms
from zamba_algorithms.pytorch_lightning.utils import (
    register_model,
    ZambaVideoClassificationLightningModule,
)
from zamba_algorithms.settings import ROOT_DIRECTORY


ASSETS_DIR = ROOT_DIRECTORY / "tests" / "assets"
TEST_VIDEOS_DIR = ASSETS_DIR / "videos"

random.seed(56745)


@register_model
class DummyZambaVideoClassificationLightningModule(ZambaVideoClassificationLightningModule):
    """A dummy model whose linear weights start out as all zeros."""

    def __init__(
        self,
        num_frames: int,
        num_hidden: int,
        finetune_from: Optional[Union[os.PathLike, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if finetune_from is None:
            backbone = torch.nn.Linear(num_frames, num_hidden)
            torch.nn.init.ones_(backbone.weight)
        else:
            backbone = self.load_from_checkpoint(finetune_from).backbone

        for param in backbone.parameters():
            param.requires_grad = False

        head = torch.nn.Linear(num_hidden, self.num_classes)
        torch.nn.init.zeros_(head.weight)

        self.backbone = backbone
        self.head = head
        self.model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten(), backbone, head
        )

        self.save_hyperparameters("num_frames", "num_hidden")

    def forward(self, x, *args, **kwargs):
        return self.model(x)


MODEL_MAPPING["dummy"] = {"transform": zamba_image_model_transforms()}


class DummyTrainConfig(TrainConfig):
    # let model name be "dummy" without causing errors
    model_name: str


@pytest.fixture
def labels_relative_path() -> os.PathLike:
    return ASSETS_DIR / "labels.csv"


@pytest.fixture
def labels_absolute_path(tmp_path) -> os.PathLike:
    output_path = tmp_path / "labels.csv"
    df = pd.read_csv(ASSETS_DIR / "labels.csv")
    df["filepath"] = (str(TEST_VIDEOS_DIR) / df.filepath.path).path.resolve()
    df.to_csv(output_path, index=False)
    return output_path


@pytest.fixture
def labels_no_splits(labels_absolute_path, tmp_path) -> os.PathLike:
    pd.read_csv(labels_absolute_path, usecols=["filepath", "label"]).to_csv(
        tmp_path / "labels_no_splits.csv", index=False
    )
    return tmp_path / "labels_no_splits.csv"


@pytest.fixture
def filepaths(labels_absolute_path, tmp_path) -> os.PathLike:
    pd.read_csv(labels_absolute_path, usecols=["filepath"]).to_csv(
        tmp_path / "filepaths.csv", index=False
    )
    return tmp_path / "filepaths.csv"


@pytest.fixture
def train_metadata(labels_absolute_path) -> pd.DataFrame:
    return TrainConfig(labels=labels_absolute_path).labels


@pytest.fixture
def predict_metadata(filepaths) -> pd.DataFrame:
    return PredictConfig(filepaths=filepaths).filepaths


@pytest.fixture
def time_distributed_checkpoint(labels_absolute_path) -> os.PathLike:
    return TrainConfig(labels=labels_absolute_path, model_name="time_distributed").checkpoint


@pytest.fixture
def mdlite():
    return MegadetectorLiteYoloX()


@pytest.fixture
def dummy_checkpoint(labels_absolute_path, tmp_path) -> os.PathLike:
    labels = pd.read_csv(labels_absolute_path)
    species = pd.get_dummies(labels.label).columns
    output_path = tmp_path / "dummy.ckpt"
    DummyZambaVideoClassificationLightningModule(
        num_frames=4, num_hidden=1, species=species
    ).to_disk(output_path)
    return output_path


@pytest.fixture
def dummy_train_config(labels_absolute_path, dummy_checkpoint, tmp_path):
    return DummyTrainConfig(
        labels=labels_absolute_path,
        data_directory=TEST_VIDEOS_DIR,
        model_name="dummy",
        checkpoint=dummy_checkpoint,
        max_epochs=1,
        batch_size=1,
        auto_lr_find=False,
        num_workers=2,
        save_directory=tmp_path / "my_model",
        skip_load_validation=True,
    )


@pytest.fixture
def dummy_video_loader_config():
    return VideoLoaderConfig(total_frames=4, frame_selection_height=19, frame_selection_width=19)


@pytest.fixture
def dummy_trainer(dummy_train_config, dummy_video_loader_config, labels_absolute_path):
    return train_model(
        train_config=dummy_train_config, video_loader_config=dummy_video_loader_config
    )


@pytest.fixture
def dummy_trained_model_checkpoint(dummy_trainer):
    # get saved out checkpoint from trainer
    return next(iter((Path(dummy_trainer.logger.log_dir).glob("*.ckpt"))))


@pytest.fixture
def caplog(_caplog):  # noqa: F811
    """Used to test logging messages from loguru per:
    https://loguru.readthedocs.io/en/stable/resources/migration.html#making-things-work-with-pytest-and-caplog
    """

    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropogateHandler(), format="{message} {extra}")
    yield _caplog
    logger.remove(handler_id)