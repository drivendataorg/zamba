import itertools

from zamba.pytorch.dataloaders import get_datasets
from zamba.pytorch_lightning.utils import ZambaDataModule


def test_get_datasets_train_metadata(train_metadata):
    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        train_metadata=train_metadata,
    )

    for video, label in itertools.chain(train_dataset, val_dataset, test_dataset):
        assert video.ndim == 4
        assert label.sum() == 1

    assert predict_dataset is None


def test_get_datasets_predict_metadata(predict_metadata):
    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        predict_metadata=predict_metadata,
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
    )

    for video, label in itertools.chain(train_dataset, val_dataset, test_dataset):
        assert video.ndim == 4
        assert label.sum() == 1

    for video, label in predict_dataset:
        assert video.ndim == 4
        assert label.sum() == 0


def test_zamba_data_module_train(train_metadata):
    data_module = ZambaDataModule(train_metadata=train_metadata)
    for videos, labels in data_module.train_dataloader():
        assert videos.ndim == 5
        assert labels.sum() == 1


def test_zamba_data_module_train_and_predict(train_metadata, predict_metadata):
    data_module = ZambaDataModule(
        train_metadata=train_metadata,
        predict_metadata=predict_metadata,
    )
    for videos, labels in data_module.train_dataloader():
        assert videos.ndim == 5
        assert labels.sum() == 1
