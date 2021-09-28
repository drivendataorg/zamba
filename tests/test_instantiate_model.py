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
        cache_dir=tmp_path,
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
        cache_dir=tmp_path,
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
        cache_dir=tmp_path,
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
        cache_dir=tmp_path,
    )
    assert scheduler_params_passed_model.hparams["scheduler_params"] == {"gamma": 0.3}


def test_remove_scheduler(time_distributed_checkpoint, tmp_path):
    """Tests that a scheduler config with None values removes the scheduler on the model."""
    remove_scheduler_model = instantiate_model(
        checkpoint=time_distributed_checkpoint,
        weight_download_region="us",
        scheduler_config=SchedulerConfig(scheduler=None),
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        cache_dir=tmp_path,
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
        cache_dir=tmp_path,
    )

    assert (model.head.weight == original_model.head.weight).all()
    assert model.hparams["species"] == [
        "species_antelope_duiker",
        "species_elephant",
        "species_gorilla",
    ]
    assert model.model[-1].out_features == 3


def test_not_predict_all_zamba_species(dummy_trained_model_checkpoint, tmp_path):
    """Tests that training a model using labels that are a subset of the model species but
    with predict_all_zamba_species=False replaces the model head."""
    original_model = DummyZambaVideoClassificationLightningModule.from_disk(
        dummy_trained_model_checkpoint
    )

    model = instantiate_model(
        checkpoint=dummy_trained_model_checkpoint,
        weight_download_region="us",
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "gorilla.mp4", "species_gorilla": 1}]),
        cache_dir=tmp_path,
        predict_all_zamba_species=False,
    )

    assert (model.head.weight != original_model.head.weight).all()
    assert model.hparams["species"] == [
        "species_gorilla",
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
        cache_dir=tmp_path,
    )

    assert (model.head.weight != original_model.head.weight).all()
    assert model.hparams["species"] == ["species_alien"]
    assert model.head.out_features == 1


@pytest.mark.parametrize("model", ["time_distributed", "slowfast", "european"])
def test_finetune_new_labels(labels_absolute_path, model, tmp_path):
    config = TrainConfig(labels=labels_absolute_path, model_name=model, skip_load_validation=True)
    model = instantiate_model(
        checkpoint=config.checkpoint,
        weight_download_region=config.weight_download_region,
        scheduler_config="default",
        labels=pd.DataFrame([{"filepath": "kangaroo.mp4", "species_kangaroo": 1}]),
        cache_dir=tmp_path,
    )
    assert model.species == ["species_kangaroo"]


@pytest.mark.parametrize("model", ["time_distributed", "slowfast", "european"])
def test_resume_subset_labels(labels_absolute_path, model, tmp_path):
    config = TrainConfig(labels=labels_absolute_path, model_name=model, skip_load_validation=True)
    model = instantiate_model(
        checkpoint=config.checkpoint,
        weight_download_region=config.weight_download_region,
        scheduler_config=SchedulerConfig(scheduler="StepLR", scheduler_params=None),
        # pick species that is present in all models
        labels=pd.DataFrame([{"filepath": "bird.mp4", "species_bird": 1}]),
        cache_dir=tmp_path,
    )
    assert model.hparams["scheduler"] == "StepLR"

    if config.model_name == "european":
        assert model.species == [
            "species_bird",
            "species_blank",
            "species_domestic_cat",
            "species_european_badger",
            "species_european_beaver",
            "species_european_hare",
            "species_european_roe_deer",
            "species_north_american_raccoon",
            "species_red_fox",
            "species_weasel",
            "species_wild_boar",
        ]
    else:
        assert model.species == [
            "species_aardvark",
            "species_antelope_duiker",
            "species_badger",
            "species_bat",
            "species_bird",
            "species_blank",
            "species_bonobo",
            "species_cattle",
            "species_cheetah",
            "species_chimpanzee",
            "species_civet_genet",
            "species_elephant",
            "species_equid",
            "species_forest_buffalo",
            "species_fox",
            "species_giraffe",
            "species_gorilla",
            "species_hare_rabbit",
            "species_hippopotamus",
            "species_hog",
            "species_human",
            "species_hyena",
            "species_large_flightless_bird",
            "species_leopard",
            "species_lion",
            "species_mongoose",
            "species_monkey_or_prosimian",
            "species_pangolin",
            "species_porcupine",
            "species_reptile",
            "species_rodent",
            "species_small_cat",
            "species_wild_dog_jackal",
        ]
