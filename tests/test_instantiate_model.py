import pandas as pd
import pytest
import torch

from zamba.models.config import SchedulerConfig, TrainConfig
from zamba.models.model_manager import instantiate_model

from conftest import DummyZambaVideoClassificationLightningModule


def test_scheduler_ignored_for_prediction(dummy_checkpoint, tmp_path):
    """Tests whether we can instantiate a model for prediction and ignore scheduler config."""
    original_hyperparams = torch.load(dummy_checkpoint)["hyper_parameters"]
    assert original_hyperparams["scheduler"] is None

    model = instantiate_model(
        checkpoint=dummy_checkpoint,
        weight_download_region="us",
        scheduler_config=SchedulerConfig(scheduler="StepLR", scheduler_params=None),
        labels=None,
        model_cache_dir=tmp_path,
    )
    # since labels is None, we are predicting. as a result, hparams are not updated
    assert model.hparams["scheduler"] is None
    # Note: using configs won't allow us to be in this situation
    # # in Train Config, which contains ModelParams, labels cannot be None


def test_default_scheduler_used(time_distributed_checkpoint, tmp_path):
    """Tests instantiate model uses the default scheduler from the hparams on the model."""
    default_scheduler_passed_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        weight_download_region="us",
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        model_cache_dir=tmp_path,
    )

    # with "default" scheduler_config, hparams from training are used
    assert default_scheduler_passed_model.hparams["scheduler"] == "MultiStepLR"
    assert default_scheduler_passed_model.hparams["scheduler_params"] == dict(
        milestones=[3], gamma=0.5, verbose=True
    )


def test_scheduler_used_if_passed(time_distributed_checkpoint, tmp_path):
    """Tests that scheduler config gets used and overrides scheduler on time distributed training."""
    scheduler_passed_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        weight_download_region="us",
        scheduler_config=SchedulerConfig(scheduler="StepLR"),
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        model_cache_dir=tmp_path,
    )

    # hparams reflect user specified scheduler config
    assert scheduler_passed_model.hparams["scheduler"] == "StepLR"
    # if no scheduler params are passed, will be None (use PTL default for that scheduler)
    assert scheduler_passed_model.hparams["scheduler_params"] is None

    # check scheduler params get used
    scheduler_params_passed_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        weight_download_region="us",
        scheduler_config=SchedulerConfig(scheduler="StepLR", scheduler_params={"gamma": 0.3}),
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        model_cache_dir=tmp_path,
    )
    assert scheduler_params_passed_model.hparams["scheduler_params"] == {"gamma": 0.3}


def test_remove_scheduler(time_distributed_checkpoint, tmp_path):
    """Tests that a scheduler config with None values removes the scheduler on the model."""
    remove_scheduler_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        weight_download_region="us",
        scheduler_config=SchedulerConfig(scheduler=None),
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        model_cache_dir=tmp_path,
    )
    # pretrained model has scheduler but this is overridden with SchedulerConfig
    assert remove_scheduler_model.hparams["scheduler"] is None


def test_head_not_replaced_for_species_subset(dummy_trained_model_checkpoint, tmp_path):
    """Tests that training a model using labels that are a subset of the model species resumes
    model training without replacing the model head."""
    original_model = DummyZambaVideoClassificationLightningModule.from_disk(
        dummy_trained_model_checkpoint
    )

    model = instantiate_model(
        checkpoint=dummy_trained_model_checkpoint,
        weight_download_region="us",
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        model_cache_dir=tmp_path,
    )

    assert (model.head.weight == original_model.head.weight).all()
    assert model.hparams["species"] == [
        "antelope_duiker",
        "elephant",
        "gorilla",
    ]
    assert model.model[-1].out_features == 3


def test_not_use_default_model_labels(dummy_trained_model_checkpoint, tmp_path):
    """Tests that training a model using labels that are a subset of the model species but
    with use_default_model_labels=False replaces the model head."""
    original_model = DummyZambaVideoClassificationLightningModule.from_disk(
        dummy_trained_model_checkpoint
    )

    model = instantiate_model(
        checkpoint=dummy_trained_model_checkpoint,
        weight_download_region="us",
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        model_cache_dir=tmp_path,
        use_default_model_labels=False,
    )

    assert (model.head.weight != original_model.head.weight).all()
    assert model.hparams["species"] == [
        "gorilla",
    ]
    assert model.model[-1].out_features == 1


def test_head_replaced_for_new_species(dummy_trained_model_checkpoint, tmp_path):
    """Tests that training a model using labels that are a not subset of the model species
    finetunes the model and replaces the model head."""
    original_model = DummyZambaVideoClassificationLightningModule.from_disk(
        dummy_trained_model_checkpoint
    )

    model = instantiate_model(
        checkpoint=dummy_trained_model_checkpoint,
        weight_download_region="us",
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "alien.mp4", "species_alien": 1}]),
        model_cache_dir=tmp_path,
    )

    assert (model.head.weight != original_model.head.weight).all()
    assert model.hparams["species"] == ["alien"]
    assert model.head.out_features == 1


@pytest.mark.parametrize("model", ["time_distributed", "slowfast", "european", "blank_nonblank"])
def test_finetune_new_labels(labels_absolute_path, model, tmp_path):
    config = TrainConfig(
        labels=labels_absolute_path,
        model_name=model,
        skip_load_validation=True,
        save_dir=tmp_path / "my_model",
    )
    model = instantiate_model(
        checkpoint=config.checkpoint,
        weight_download_region=config.weight_download_region,
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "kangaroo.mp4", "species_kangaroo": 1}]),
        model_cache_dir=tmp_path,
    )
    assert model.species == ["kangaroo"]


@pytest.mark.parametrize("model", ["time_distributed", "slowfast", "european"])
def test_resume_subset_labels(labels_absolute_path, model, tmp_path):
    # note: there are no additional species to add for the blank_nonblank model so it is not tested
    config = TrainConfig(
        labels=labels_absolute_path,
        model_name=model,
        skip_load_validation=True,
        save_dir=tmp_path / "my_model",
    )
    model = instantiate_model(
        checkpoint=config.checkpoint,
        weight_download_region=config.weight_download_region,
        scheduler_config=SchedulerConfig(scheduler="StepLR", scheduler_params=None),
        # pick species that is present in all models
        labels=pd.DataFrame([{"filepath": "bird.mp4", "species_bird": 1}]),
        model_cache_dir=tmp_path,
    )
    assert model.hparams["scheduler"] == "StepLR"

    if config.model_name == "european":
        assert model.species == [
            "bird",
            "blank",
            "domestic_cat",
            "european_badger",
            "european_beaver",
            "european_hare",
            "european_roe_deer",
            "north_american_raccoon",
            "red_fox",
            "weasel",
            "wild_boar",
        ]
    else:
        assert model.species == [
            "aardvark",
            "antelope_duiker",
            "badger",
            "bat",
            "bird",
            "blank",
            "cattle",
            "cheetah",
            "chimpanzee_bonobo",
            "civet_genet",
            "elephant",
            "equid",
            "forest_buffalo",
            "fox",
            "giraffe",
            "gorilla",
            "hare_rabbit",
            "hippopotamus",
            "hog",
            "human",
            "hyena",
            "large_flightless_bird",
            "leopard",
            "lion",
            "mongoose",
            "monkey_prosimian",
            "pangolin",
            "porcupine",
            "reptile",
            "rodent",
            "small_cat",
            "wild_dog_jackal",
        ]
