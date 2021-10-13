import json
from pathlib import Path
import yaml

from botocore.exceptions import ClientError
import pytest
import torch

from zamba.models.config import MODEL_MAPPING
from zamba.models.utils import download_weights
from zamba.models.model_manager import train_model

from conftest import DummyTrainConfig, TEST_VIDEOS_DIR, labels_n_classes_df


def test_model_manager(dummy_trainer):
    assert (dummy_trainer.model.model[2].weight == 1).all()
    assert not (dummy_trainer.model.model[3].weight == 0).all()


def test_save_checkpoint(dummy_trained_model_checkpoint):
    checkpoint = torch.load(dummy_trained_model_checkpoint)

    assert tuple(checkpoint["state_dict"].keys()) == (
        "backbone.weight",
        "backbone.bias",
        "head.weight",
        "head.bias",
        "model.2.weight",
        "model.2.bias",
        "model.3.weight",
        "model.3.bias",
    )

    assert checkpoint["hyper_parameters"] == {
        "lr": 0.001,
        "model_class": "DummyZambaVideoClassificationLightningModule",
        "num_frames": 4,
        "num_hidden": 1,
        "scheduler": None,
        "scheduler_params": None,
        "species": ["antelope_duiker", "elephant", "gorilla"],
    }


@pytest.mark.parametrize("split", ("test", "val"))
def test_save_metrics(dummy_trainer, split):
    metric_names = {
        "val_loss",
        f"{split}_macro_f1",
        f"{split}_top_1_accuracy",
        f"species/{split}_accuracy/antelope_duiker",
        f"species/{split}_f1/antelope_duiker",
        f"species/{split}_precision/antelope_duiker",
        f"species/{split}_recall/antelope_duiker",
        f"species/{split}_accuracy/elephant",
        f"species/{split}_f1/elephant",
        f"species/{split}_precision/elephant",
        f"species/{split}_recall/elephant",
        f"species/{split}_accuracy/gorilla",
        f"species/{split}_f1/gorilla",
        f"species/{split}_precision/gorilla",
        f"species/{split}_recall/gorilla",
    }

    with (Path(dummy_trainer.logger.log_dir) / f"{split}_metrics.json").open() as fp:
        metrics = json.load(fp)
    assert metrics.keys() == metric_names


@pytest.mark.parametrize("split", ("test", "val"))
def test_save_metrics_less_than_two_classes(
    dummy_video_loader_config, split, dummy_checkpoint, tmp_path
):
    labels = labels_n_classes_df(2)
    trainer = train_model(
        train_config=DummyTrainConfig(
            labels=labels,
            data_directory=TEST_VIDEOS_DIR,
            model_name="dummy",
            checkpoint=dummy_checkpoint,
            max_epochs=1,
            batch_size=1,
            auto_lr_find=False,
            num_workers=2,
            save_directory=tmp_path / "my_model",
            skip_load_validation=True,
        ),
        video_loader_config=dummy_video_loader_config,
    )

    with (Path(trainer.logger.log_dir) / f"{split}_metrics.json").open() as fp:
        metrics = json.load(fp)

    metric_names = {
        "val_loss",
        f"{split}_macro_f1",
        f"{split}_accuracy",
    }

    for c in labels.label.unique():
        metric_names = metric_names.union(
            {
                f"species/{split}_accuracy/{c}",
                f"species/{split}_f1/{c}",
                f"species/{split}_precision/{c}",
                f"species/{split}_recall/{c}",
            }
        )

    assert metrics.keys() == metric_names


def test_save_configuration(dummy_trainer):
    with (Path(dummy_trainer.logger.log_dir) / "train_configuration.yaml").open() as fp:
        config = yaml.safe_load(fp.read())

    assert set(config.keys()) == {
        "git_hash",
        "model_class",
        "species",
        "starting_learning_rate",
        "train_config",
        "training_start_time",
        "video_loader_config",
    }


def test_train_save_directory(dummy_trainer):
    assert Path(dummy_trainer.logger.root_dir).name == "my_model"
    assert Path(dummy_trainer.logger.log_dir).name == "version_0"


def test_train_save_directory_overwrite(
    labels_absolute_path, dummy_checkpoint, tmp_path, dummy_video_loader_config
):
    config = DummyTrainConfig(
        labels=labels_absolute_path,
        data_directory=TEST_VIDEOS_DIR,
        model_name="dummy",
        checkpoint=dummy_checkpoint,
        save_directory=tmp_path / "my_model",
        skip_load_validation=True,
        overwrite_save_directory=True,
        max_epochs=1,
        batch_size=1,
        auto_lr_find=False,
        num_workers=1,
    )

    overwrite_trainer = train_model(
        train_config=config, video_loader_config=dummy_video_loader_config
    )

    assert Path(overwrite_trainer.logger.log_dir).resolve() == config.save_directory.resolve()

    assert not any([f.name.startswith("version_") for f in config.save_directory.iterdir()])

    for f in ["train_configuration.yaml", "test_metrics.json", "val_metrics.json", "dummy.ckpt"]:
        assert (config.save_directory / f).exists()


@pytest.mark.parametrize("model_name", ["time_distributed", "slowfast", "european"])
@pytest.mark.parametrize("weight_region", ["us", "asia", "eu"])
def test_download_weights(model_name, weight_region, tmp_path):
    public_weights = MODEL_MAPPING[model_name]["public_weights"]
    if weight_region != "us":
        region_bucket = f"drivendata-public-assets-{weight_region}"
    else:
        region_bucket = "drivendata-public-assets"
    fspath = download_weights(
        filename=public_weights,
        weight_region=weight_region,
        destination_dir=tmp_path,
    )
    # ensure weights exist
    assert Path(fspath).exists()
    # ensure path is correct
    assert Path(fspath) == tmp_path / region_bucket / public_weights

    # invalid filename
    with pytest.raises(ClientError):
        download_weights(
            filename="incorrect_checkpoint.ckpt", destination_dir=tmp_path, weight_region="us"
        )