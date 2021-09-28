import json
from pathlib import Path
import yaml

from botocore.exceptions import ClientError
import pytest
import torch

from zamba.models.config import MODEL_MAPPING
from zamba.models.utils import download_weights


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
        "species": ["species_antelope_duiker", "species_elephant", "species_gorilla"],
    }


@pytest.mark.parametrize("split", ("test", "val"))
def test_save_metrics(dummy_trainer, split):
    metric_names = {
        "val_loss",
        f"{split}_macro_f1",
        f"{split}_top_1_accuracy",
        f"species/{split}_accuracy/species_antelope_duiker",
        f"species/{split}_f1/species_antelope_duiker",
        f"species/{split}_precision/species_antelope_duiker",
        f"species/{split}_recall/species_antelope_duiker",
        f"species/{split}_accuracy/species_elephant",
        f"species/{split}_f1/species_elephant",
        f"species/{split}_precision/species_elephant",
        f"species/{split}_recall/species_elephant",
        f"species/{split}_accuracy/species_gorilla",
        f"species/{split}_f1/species_gorilla",
        f"species/{split}_precision/species_gorilla",
        f"species/{split}_recall/species_gorilla",
    }

    with (Path(dummy_trainer.logger.log_dir) / f"{split}_metrics.json").open() as fp:
        metrics = json.load(fp)
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
