import itertools
import os

import pytest

from zamba_algorithms.pytorch.dataloaders import get_datasets
from zamba_algorithms.pytorch_lightning.utils import ZambaDataModule


def test_get_datasets_train_metadata(train_metadata):
    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        train_metadata=train_metadata,
        load_metadata_config=None,
    )

    for video, label in itertools.chain(train_dataset, val_dataset, test_dataset):
        assert video.ndim == 4
        assert label.sum() == 1

    assert predict_dataset is None


def test_get_datasets_predict_metadata(predict_metadata):
    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        predict_metadata=predict_metadata,
        load_metadata_config=None,
    )

    for video, label in predict_dataset:
        assert video.ndim == 4
        assert label.sum() == 0

    assert train_dataset is None
    assert val_dataset is None
    assert test_dataset is None


def test_get_datasets_train_and_predict_metadata(train_metadata, predict_metadata):
    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        train_metadata=train_metadata,
        predict_metadata=predict_metadata,
        load_metadata_config=None,
    )

    for video, label in itertools.chain(train_dataset, val_dataset, test_dataset):
        assert video.ndim == 4
        assert label.sum() == 1

    for video, label in predict_dataset:
        assert video.ndim == 4
        assert label.sum() == 0


@pytest.mark.skipif(bool(os.getenv("CI")), reason="Private metadata files not available in CI")
def test_get_datasets_load_metadata():
    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        load_metadata_config={"subset": "dev", "zamba_label": "new"}
    )
    assert len(train_dataset) == 2577
    assert len(val_dataset) == 2404
    assert len(test_dataset) == 2326
    assert predict_dataset is None


def test_zamba_data_module_train(train_metadata):
    data_module = ZambaDataModule(train_metadata=train_metadata, load_metadata_config=None)
    for videos, labels in data_module.train_dataloader():
        assert videos.ndim == 5
        assert labels.sum() == 1


def test_zamba_data_module_train_and_predict(train_metadata, predict_metadata):
    data_module = ZambaDataModule(
        train_metadata=train_metadata,
        predict_metadata=predict_metadata,
        load_metadata_config=None,
    )
    for videos, labels in data_module.train_dataloader():
        assert videos.ndim == 5
        assert labels.sum() == 1
