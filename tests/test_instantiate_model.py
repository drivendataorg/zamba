import pandas as pd
import pytest
import torch

from zamba.models.config import SchedulerConfig, TrainConfig
from zamba.models.model_manager import instantiate_model
from zamba.models.utils import get_model_species

from conftest import DummyZambaVideoClassificationLightningModule


def test_scheduler_ignored_for_prediction(dummy_checkpoint):
    """Tests whether we can instantiate a model for prediction and ignore scheduler config."""
    original_hyperparams = torch.load(dummy_checkpoint)["hyper_parameters"]
    assert original_hyperparams["scheduler"] is None

    model = instantiate_model(
        checkpoint=dummy_checkpoint,
        scheduler_config=SchedulerConfig(scheduler="StepLR", scheduler_params=None),
        labels=None,
    )
    # since labels is None, we are predicting. as a result, hparams are not updated
    assert model.hparams["scheduler"] is None
    # Note: using configs won't allow us to be in this situation
    # # in Train Config, which contains ModelParams, labels cannot be None


def test_default_scheduler_used(time_distributed_checkpoint):
    """Tests instantiate model uses the default scheduler from the hparams on the model."""
    default_scheduler_passed_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
    )

    # with "default" scheduler_config, hparams from training are used
    assert default_scheduler_passed_model.hparams["scheduler"] == "MultiStepLR"
    assert default_scheduler_passed_model.hparams["scheduler_params"] == dict(
        milestones=[3], gamma=0.5, verbose=True
    )


def test_scheduler_used_if_passed(time_distributed_checkpoint):
    """Tests that scheduler config gets used and overrides scheduler on time distributed training."""
    scheduler_passed_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        scheduler_config=SchedulerConfig(scheduler="StepLR"),
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
    )

    # hparams reflect user specified scheduler config
    assert scheduler_passed_model.hparams["scheduler"] == "StepLR"
    # if no scheduler params are passed, will be None (use PTL default for that scheduler)
    assert scheduler_passed_model.hparams["scheduler_params"] is None

    # check scheduler params get used
    scheduler_params_passed_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        scheduler_config=SchedulerConfig(scheduler="StepLR", scheduler_params={"gamma": 0.3}),
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
    )
    assert scheduler_params_passed_model.hparams["scheduler_params"] == {"gamma": 0.3}


def test_remove_scheduler(time_distributed_checkpoint):
    """Tests that a scheduler config with None values removes the scheduler on the model."""
    remove_scheduler_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        scheduler_config=SchedulerConfig(scheduler=None),
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
    )
    # pretrained model has scheduler but this is overridden with SchedulerConfig
    assert remove_scheduler_model.hparams["scheduler"] is None


def test_use_default_model_labels(dummy_trained_model_checkpoint):
    """Tests that training a model using labels that are a subset of the model species resumes
    model training without replacing the model head."""
    original_model = DummyZambaVideoClassificationLightningModule.from_disk(
        dummy_trained_model_checkpoint
    )

    model = instantiate_model(
        checkpoint=dummy_trained_model_checkpoint,
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        use_default_model_labels=True,
    )

    assert (model.head.weight == original_model.head.weight).all()
    assert model.hparams["species"] == [
        "antelope_duiker",
        "elephant",
        "gorilla",
    ]
    assert model.model[-1].out_features == 3


def test_not_use_default_model_labels(dummy_trained_model_checkpoint):
    """Tests that training a model using labels that are a subset of the model species but
    with use_default_model_labels=False replaces the model head."""
    original_model = DummyZambaVideoClassificationLightningModule.from_disk(
        dummy_trained_model_checkpoint
    )

    model = instantiate_model(
        checkpoint=dummy_trained_model_checkpoint,
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        use_default_model_labels=False,
    )

    assert (model.head.weight != original_model.head.weight).all()
    assert model.hparams["species"] == [
        "gorilla",
    ]
    assert model.model[-1].out_features == 1


@pytest.mark.parametrize(
    "model_name", ["time_distributed", "slowfast", "european", "blank_nonblank"]
)
def test_head_replaced_for_new_species(labels_absolute_path, model_name, tmp_path):
    """Check that output species reflect the new head."""
    # pick species that is not present in any models
    labels = pd.read_csv(labels_absolute_path)
    labels["label"] = "kangaroo"

    config = TrainConfig(
        labels=labels,
        model_name=model_name,
        skip_load_validation=True,
        save_dir=tmp_path / "my_model",
    )
    model = instantiate_model(
        checkpoint=config.checkpoint,
        scheduler_config="default",
        labels=config.labels,
        use_default_model_labels=config.use_default_model_labels,
    )

    assert model.hparams["species"] == model.species == ["kangaroo"]


@pytest.mark.parametrize("model_name", ["time_distributed", "slowfast", "european"])
def test_resume_subset_labels(labels_absolute_path, model_name, tmp_path):
    """Check that output species reflect the default model labels."""
    # pick species that is present in all models
    labels = pd.read_csv(labels_absolute_path)
    labels["label"] = "bird"

    config = TrainConfig(
        labels=labels,
        model_name=model_name,
        skip_load_validation=True,
        save_dir=tmp_path / "my_model",
    )
    model = instantiate_model(
        checkpoint=config.checkpoint,
        scheduler_config=SchedulerConfig(scheduler="StepLR", scheduler_params=None),
        labels=config.labels,
        use_default_model_labels=config.use_default_model_labels,
    )
    assert model.hparams["scheduler"] == "StepLR"
    assert model.species == get_model_species(checkpoint=None, model_name=model_name)
