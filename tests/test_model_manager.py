import json
from pathlib import Path

from botocore.exceptions import ClientError
import pytest
import torch
import yaml

from zamba.models.utils import download_weights, get_model_checkpoint_filename
from zamba.models.model_manager import train_model

from conftest import DummyTrainConfig, TEST_VIDEOS_DIR, labels_n_classes_df


def test_model_manager(dummy_trainer):
    assert (dummy_trainer.model.model[2].weight == 1).all()
    assert not (dummy_trainer.model.model[3].weight == 0).all()


def test_no_early_stopping(
    labels_absolute_path, tmp_path, dummy_checkpoint, dummy_video_loader_config
):
    config = DummyTrainConfig(
        labels=labels_absolute_path,
        data_dir=TEST_VIDEOS_DIR,
        checkpoint=dummy_checkpoint,
        early_stopping_config=None,
        save_dir=tmp_path / "my_model",
        num_workers=1,
    )
    train_model(train_config=config, video_loader_config=dummy_video_loader_config)


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
            data_dir=TEST_VIDEOS_DIR,
            checkpoint=dummy_checkpoint,
            num_workers=2,
            save_dir=tmp_path / "my_model",
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

    for c in labels.label.str.lower().unique():
        metric_names = metric_names.union(
            {
                f"species/{split}_accuracy/{c}",
                f"species/{split}_f1/{c}",
                f"species/{split}_precision/{c}",
                f"species/{split}_recall/{c}",
            }
        )

    removed_in_binary_case = {
        "species/test_precision/b",
        "species/test_recall/b",
        "species/test_accuracy/b",
        "species/test_f1/b",
        "species/val_precision/b",
        "species/val_recall/b",
        "species/val_accuracy/b",
        "species/val_f1/b",
    }
    assert metrics.keys() == metric_names - removed_in_binary_case


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


def test_train_save_dir(dummy_trainer):
    assert Path(dummy_trainer.logger.root_dir).name == "my_model"
    assert Path(dummy_trainer.logger.log_dir).name == "version_0"


def test_train_save_dir_overwrite(
    labels_absolute_path, dummy_checkpoint, tmp_path, dummy_video_loader_config
):
    config = DummyTrainConfig(
        labels=labels_absolute_path,
        data_dir=TEST_VIDEOS_DIR,
        checkpoint=dummy_checkpoint,
        save_dir=tmp_path / "my_model",
        overwrite=True,
        num_workers=1,
    )

    overwrite_trainer = train_model(
        train_config=config, video_loader_config=dummy_video_loader_config
    )

    assert Path(overwrite_trainer.logger.log_dir).resolve() == config.save_dir.resolve()

    assert not any([f.name.startswith("version_") for f in config.save_dir.iterdir()])

    for f in [
        "train_configuration.yaml",
        "test_metrics.json",
        "val_metrics.json",
        "dummy_model.ckpt",
    ]:
        assert (config.save_dir / f).exists()


@pytest.mark.parametrize(
    "model_name", ["time_distributed", "slowfast", "european", "blank_nonblank"]
)
@pytest.mark.parametrize("weight_region", ["us", "asia", "eu"])
def test_download_weights(model_name, weight_region, tmp_path):
    public_weights = get_model_checkpoint_filename(model_name)

    ckpt_path = download_weights(
        filename=public_weights,
        weight_region=weight_region,
        destination_dir=tmp_path,
    )
    # ensure download happened
    assert Path(ckpt_path).exists()

    # ensure path is correct
    assert Path(ckpt_path) == tmp_path / public_weights

    # invalid filename
    with pytest.raises(ClientError):
        download_weights(
            filename="incorrect_checkpoint.ckpt", destination_dir=tmp_path, weight_region="us"
        )
