import logging
import os
import random
import string
from typing import Optional, Union

from loguru import logger
import pandas as pd
from pathlib import Path
import pytest
from _pytest.logging import caplog as _caplog  # noqa: F401
import torch

from zamba.data.video import VideoLoaderConfig
from zamba.models.config import PredictConfig, TrainConfig
from zamba.models.model_manager import MODEL_MAPPING, train_model
from zamba.models.registry import register_model
from zamba.object_detection.yolox.megadetector_lite_yolox import MegadetectorLiteYoloX
from zamba.pytorch.transforms import zamba_image_model_transforms
from zamba.pytorch_lightning.utils import ZambaVideoClassificationLightningModule


ASSETS_DIR = Path(__file__).parent / "assets"
TEST_VIDEOS_DIR = ASSETS_DIR / "videos"

random.seed(56745)


@register_model
class DummyZambaVideoClassificationLightningModule(ZambaVideoClassificationLightningModule):
    """A dummy model whose linear weights start out as all zeros."""

    _default_model_name = "dummy_model"  # used to look up default configuration for checkpoints

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
            backbone = self.from_disk(finetune_from).backbone

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


MODEL_MAPPING["DummyZambaVideoClassificationLightningModule"] = {
    "transform": zamba_image_model_transforms()
}


class DummyTrainConfig(TrainConfig):
    # let model name be "dummy" without causing errors
    model_name: str
    batch_size = 1
    max_epochs = 1
    model_name = "dummy"
    skip_load_validation = True
    auto_lr_find = False


@pytest.fixture(scope="session")
def labels_relative_path() -> os.PathLike:
    return ASSETS_DIR / "labels.csv"


def labels_n_classes_df(n_classes):
    """Get up a labels dataframe where the labels are up to
    26 classes.
    """
    if n_classes > len(string.ascii_uppercase):
        raise ValueError("n_classes must be less than 26")

    choices = string.ascii_uppercase[:n_classes]

    df = pd.read_csv(ASSETS_DIR / "labels.csv")
    df.label = random.choices(choices, k=len(df))

    return df


@pytest.fixture(scope="session")
def labels_absolute_path(tmp_path_factory) -> os.PathLike:
    tmp_path = tmp_path_factory.mktemp("dummy-model-dir")
    output_path = tmp_path / "labels.csv"
    df = pd.read_csv(ASSETS_DIR / "labels.csv")
    df["filepath"] = (str(TEST_VIDEOS_DIR) / df.filepath.path).path.resolve()
    df.to_csv(output_path, index=False)
    return output_path


@pytest.fixture(scope="session")
def labels_no_splits(labels_absolute_path, tmp_path_factory) -> os.PathLike:
    tmp_path = tmp_path_factory.mktemp("dummy-model-dir")
    pd.read_csv(labels_absolute_path, usecols=["filepath", "label"]).to_csv(
        tmp_path / "labels_no_splits.csv", index=False
    )
    return tmp_path / "labels_no_splits.csv"


@pytest.fixture(scope="session")
def filepaths(labels_absolute_path, tmp_path_factory) -> os.PathLike:
    tmp_path = tmp_path_factory.mktemp("dummy-model-dir")
    pd.read_csv(labels_absolute_path, usecols=["filepath"]).to_csv(
        tmp_path / "filepaths.csv", index=False
    )
    return tmp_path / "filepaths.csv"


@pytest.fixture(scope="session")
def train_metadata(labels_absolute_path) -> pd.DataFrame:
    return TrainConfig(labels=labels_absolute_path).labels


@pytest.fixture(scope="session")
def predict_metadata(filepaths) -> pd.DataFrame:
    return PredictConfig(filepaths=filepaths).filepaths


@pytest.fixture(scope="session")
def time_distributed_checkpoint(labels_absolute_path) -> os.PathLike:
    return TrainConfig(labels=labels_absolute_path, model_name="time_distributed").checkpoint


@pytest.fixture(scope="session")
def mdlite():
    return MegadetectorLiteYoloX()


@pytest.fixture(scope="session")
def dummy_checkpoint(labels_absolute_path, tmp_path_factory) -> os.PathLike:
    tmp_path = tmp_path_factory.mktemp("dummy-model-dir")
    labels = pd.read_csv(labels_absolute_path)
    species = list(labels.label.unique())
    output_path = tmp_path / "dummy.ckpt"
    DummyZambaVideoClassificationLightningModule(
        num_frames=4, num_hidden=1, species=species
    ).to_disk(output_path)
    return output_path


@pytest.fixture(scope="session")
def dummy_train_config(labels_absolute_path, dummy_checkpoint, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dummy-model-dir")
    return DummyTrainConfig(
        labels=labels_absolute_path,
        data_dir=TEST_VIDEOS_DIR,
        model_name="dummy",
        checkpoint=dummy_checkpoint,
        max_epochs=1,
        batch_size=1,
        auto_lr_find=False,
        num_workers=2,
        save_dir=tmp_path / "my_model",
        skip_load_validation=True,
    )


@pytest.fixture(scope="session")
def dummy_video_loader_config():
    return VideoLoaderConfig(total_frames=4, frame_selection_height=19, frame_selection_width=19)


@pytest.fixture(scope="session")
def dummy_trainer(dummy_train_config, dummy_video_loader_config):
    return train_model(
        train_config=dummy_train_config, video_loader_config=dummy_video_loader_config
    )


@pytest.fixture(scope="session")
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
