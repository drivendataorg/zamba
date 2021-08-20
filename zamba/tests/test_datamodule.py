import itertools
import random
import pandas as pd
import pytest

from zamba_algorithms.pytorch.dataloaders import get_datasets
from zamba_algorithms.pytorch_lightning.utils import ZambaDataModule
from zamba_algorithms.settings import ROOT_DIRECTORY

TEST_DATA_DIRECTORY = ROOT_DIRECTORY / "tests" / "assets" / "videos"
random.seed(56745)


@pytest.fixture
def train_metadata_no_splits(tmp_path):
    pd.DataFrame(
        [
            {
                "filepath": path.relative_to(TEST_DATA_DIRECTORY),
                "label": random.choice(["bird", "dog", "cat"]),
            }
            for path in TEST_DATA_DIRECTORY.rglob("*")
            if path.is_file()
        ]
    ).to_csv(tmp_path / "train_no_splits.csv", index=False)
    return "train_no_splits.csv"


@pytest.fixture
def train_metadata_with_splits(tmp_path):
    pd.DataFrame(
        [
            {
                "filepath": path.relative_to(TEST_DATA_DIRECTORY),
                "label": random.choice(["bird", "dog", "cat"]),
                "split": random.choice(["train", "val", "holdout"]),
            }
            for path in TEST_DATA_DIRECTORY.rglob("*")
            if path.is_file()
        ]
    ).to_csv(tmp_path / "train_with_splits.csv", index=False)
    return "train_with_splits.csv"


@pytest.fixture
def predict_metadata(tmp_path):
    pd.DataFrame(
        [
            {"filepath": path.relative_to(TEST_DATA_DIRECTORY)}
            for path in TEST_DATA_DIRECTORY.rglob("*")
            if path.is_file()
        ]
    ).to_csv(tmp_path / "predict.csv", index=False)
    return "predict.csv"


@pytest.mark.parametrize("preload_metadata", (True, False))
def test_get_datasets_train_metadata_no_splits(
    train_metadata_no_splits, tmp_path, preload_metadata
):
    train_metadata = (
        pd.read_csv(tmp_path / train_metadata_no_splits)
        if preload_metadata
        else tmp_path / train_metadata_no_splits
    )

    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        train_metadata=train_metadata, load_metadata_config=None, video_dir=TEST_DATA_DIRECTORY
    )

    for video, label in itertools.chain(train_dataset, val_dataset, test_dataset):
        assert video.ndim == 4
        assert label.sum() == 1

    assert predict_dataset is None


@pytest.mark.parametrize("preload_metadata", (True, False))
def test_get_datasets_train_metadata_with_splits(
    train_metadata_with_splits, tmp_path, preload_metadata
):
    train_metadata = (
        pd.read_csv(tmp_path / train_metadata_with_splits)
        if preload_metadata
        else tmp_path / train_metadata_with_splits
    )

    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        train_metadata=train_metadata,
        load_metadata_config=None,
        video_dir=TEST_DATA_DIRECTORY,
    )

    for video, label in itertools.chain(train_dataset, val_dataset, test_dataset):
        assert video.ndim == 4
        assert label.sum() == 1

    assert predict_dataset is None


@pytest.mark.parametrize("preload_metadata", (True, False))
def test_get_datasets_predict_metadata(predict_metadata, tmp_path, preload_metadata):
    predict_metadata = (
        pd.read_csv(tmp_path / predict_metadata)
        if preload_metadata
        else tmp_path / predict_metadata
    )

    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        predict_metadata=predict_metadata,
        load_metadata_config=None,
        video_dir=TEST_DATA_DIRECTORY,
    )

    for video, label in predict_dataset:
        assert video.ndim == 4
        assert label.sum() == 0

    assert train_dataset is None
    assert val_dataset is None
    assert test_dataset is None


@pytest.mark.parametrize("preload_metadata", (True, False))
def test_get_datasets_train_and_predict_metadata(
    train_metadata_no_splits, predict_metadata, tmp_path, preload_metadata
):
    train_metadata = (
        pd.read_csv(tmp_path / train_metadata_no_splits)
        if preload_metadata
        else tmp_path / train_metadata_no_splits
    )

    predict_metadata = (
        pd.read_csv(tmp_path / predict_metadata)
        if preload_metadata
        else tmp_path / predict_metadata
    )

    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        train_metadata=train_metadata,
        predict_metadata=predict_metadata,
        load_metadata_config=None,
        video_dir=TEST_DATA_DIRECTORY,
    )

    for video, label in itertools.chain(train_dataset, val_dataset, test_dataset):
        assert video.ndim == 4
        assert label.sum() == 1

    for video, label in predict_dataset:
        assert video.ndim == 4
        assert label.sum() == 0


@pytest.mark.parametrize("preload_metadata", (True, False))
def test_get_datasets_split_proportions(train_metadata_no_splits, tmp_path, preload_metadata):
    train_metadata = (
        pd.read_csv(tmp_path / train_metadata_no_splits)
        if preload_metadata
        else tmp_path / train_metadata_no_splits
    )

    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        train_metadata=train_metadata,
        split_proportions={"train": 1, "val": 3, "holdout": 1},
        load_metadata_config=None,
        video_dir=TEST_DATA_DIRECTORY,
    )
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0
    assert predict_dataset is None
    assert len(train_dataset) < len(val_dataset)


def test_get_datasets_load_metadata():
    train_dataset, val_dataset, test_dataset, predict_dataset = get_datasets(
        load_metadata_config={"subset": "dev"}, video_dir=TEST_DATA_DIRECTORY
    )
    assert len(train_dataset) == 1851
    assert len(val_dataset) == 1927
    assert len(test_dataset) == 1802
    assert predict_dataset is None


@pytest.mark.parametrize("preload_metadata", (True, False))
def test_zamba_data_module_train(train_metadata_with_splits, tmp_path, preload_metadata):
    train_metadata = (
        pd.read_csv(tmp_path / train_metadata_with_splits)
        if preload_metadata
        else tmp_path / train_metadata_with_splits
    )

    data_module = ZambaDataModule(
        TEST_DATA_DIRECTORY, train_metadata=train_metadata, load_metadata_config=None
    )
    for videos, labels in data_module.train_dataloader():
        assert videos.ndim == 5
        assert labels.sum() == 1


@pytest.mark.parametrize("preload_metadata", (True, False))
def test_zamba_data_module_train_and_predict(
    train_metadata_with_splits, predict_metadata, tmp_path, preload_metadata
):
    train_metadata = (
        pd.read_csv(tmp_path / train_metadata_with_splits)
        if preload_metadata
        else tmp_path / train_metadata_with_splits
    )

    predict_metadata = (
        pd.read_csv(tmp_path / predict_metadata)
        if preload_metadata
        else tmp_path / predict_metadata
    )

    data_module = ZambaDataModule(
        TEST_DATA_DIRECTORY,
        train_metadata=train_metadata,
        predict_metadata=predict_metadata,
        load_metadata_config=None,
    )
    for videos, labels in data_module.train_dataloader():
        assert videos.ndim == 5
        assert labels.sum() == 1
