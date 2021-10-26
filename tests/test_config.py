import os
from pathlib import Path
import pytest

import appdirs
import numpy as np
import pandas as pd
from pydantic import ValidationError

from zamba.models.config import (
    EarlyStoppingConfig,
    ModelConfig,
    PredictConfig,
    SchedulerConfig,
    TrainConfig,
)

from conftest import ASSETS_DIR, TEST_VIDEOS_DIR


def test_train_data_dir_only():
    with pytest.raises(ValidationError) as error:
        TrainConfig(data_dir=TEST_VIDEOS_DIR)
    # labels is missing
    assert error.value.errors() == [
        {"loc": ("labels",), "msg": "field required", "type": "value_error.missing"}
    ]


def test_train_data_dir_and_labels(tmp_path, labels_relative_path, labels_absolute_path):
    # correct data dir
    config = TrainConfig(data_dir=TEST_VIDEOS_DIR, labels=labels_relative_path)
    assert config.data_dir is not None
    assert config.labels is not None

    # data dir ignored if absolute path provided in filepath
    config = TrainConfig(data_dir=tmp_path, labels=labels_absolute_path)
    assert config.data_dir is not None
    assert config.labels is not None
    assert not config.labels.filepath.str.startswith(str(tmp_path)).any()

    # incorrect data dir with relative filepaths
    with pytest.raises(ValidationError) as error:
        TrainConfig(data_dir=ASSETS_DIR, labels=labels_relative_path)
    assert "None of the video filepaths exist" in error.value.errors()[0]["msg"]


def test_train_labels_only(labels_absolute_path):
    config = TrainConfig(labels=labels_absolute_path)
    assert config.labels is not None


def test_predict_data_dir_only():
    config = PredictConfig(data_dir=TEST_VIDEOS_DIR)
    assert config.data_dir == TEST_VIDEOS_DIR
    assert isinstance(config.filepaths, pd.DataFrame)
    assert sorted(config.filepaths.filepath.values) == sorted(
        [str(f) for f in TEST_VIDEOS_DIR.rglob("*") if f.is_file()]
    )
    assert config.filepaths.columns == ["filepath"]


def test_predict_data_dir_and_filepaths(labels_absolute_path, labels_relative_path):
    # correct data dir
    config = PredictConfig(data_dir=TEST_VIDEOS_DIR, filepaths=labels_relative_path)
    assert config.data_dir is not None
    assert config.filepaths is not None
    assert config.filepaths.filepath.str.startswith(str(TEST_VIDEOS_DIR)).all()

    # incorrect data dir
    with pytest.raises(ValidationError) as error:
        PredictConfig(data_dir=ASSETS_DIR, filepaths=labels_relative_path)
    assert "None of the video filepaths exist" in error.value.errors()[0]["msg"]


def test_predict_filepaths_only(labels_absolute_path):
    config = PredictConfig(filepaths=labels_absolute_path)
    assert config.filepaths is not None


def test_filepath_column(tmp_path, labels_absolute_path):
    pd.read_csv(labels_absolute_path).rename(columns={"filepath": "video"}).to_csv(
        tmp_path / "bad_filepath_column.csv"
    )
    # predict: filepaths
    with pytest.raises(ValidationError) as error:
        PredictConfig(filepaths=tmp_path / "bad_filepath_column.csv")
    assert "must contain a `filepath` column" in error.value.errors()[0]["msg"]

    # train: labels
    with pytest.raises(ValidationError) as error:
        TrainConfig(labels=tmp_path / "bad_filepath_column.csv")
    assert "must contain `filepath` and `label` columns" in error.value.errors()[0]["msg"]


def test_label_column(tmp_path, labels_absolute_path):
    pd.read_csv(labels_absolute_path).rename(columns={"label": "animal"}).to_csv(
        tmp_path / "bad_label_column.csv"
    )
    with pytest.raises(ValidationError) as error:
        TrainConfig(labels=tmp_path / "bad_label_column.csv")
    assert "must contain `filepath` and `label` columns" in error.value.errors()[0]["msg"]


def test_one_video_does_not_exist(tmp_path, labels_absolute_path, caplog):
    files_df = pd.read_csv(labels_absolute_path)
    # add a fake file
    files_df = files_df.append(
        {"filepath": "fake_file.mp4", "label": "gorilla", "split": "train"}, ignore_index=True
    )
    files_df.to_csv(tmp_path / "labels_with_fake_video.csv")

    config = PredictConfig(filepaths=tmp_path / "labels_with_fake_video.csv")
    assert "Skipping 1 file(s) that could not be found" in caplog.text
    # one fewer file than in original list since bad file is skipped
    assert len(config.filepaths) == (len(files_df) - 1)

    config = TrainConfig(labels=tmp_path / "labels_with_fake_video.csv")
    assert "Skipping 1 file(s) that could not be found" in caplog.text
    assert len(config.labels) == (len(files_df) - 1)


def test_videos_cannot_be_loaded(tmp_path, labels_absolute_path, caplog):
    files_df = pd.read_csv(labels_absolute_path)
    # create bad files
    for i in np.arange(2):
        bad_file = tmp_path / f"bad_file_{i}.mp4"
        bad_file.touch()
        files_df = files_df.append(
            {"filepath": bad_file, "label": "gorilla", "split": "train"}, ignore_index=True
        )

    files_df.to_csv(tmp_path / "labels_with_non_loadable_videos.csv")

    config = PredictConfig(filepaths=tmp_path / "labels_with_non_loadable_videos.csv")
    assert "Skipping 2 file(s) that could not be loaded with ffmpeg" in caplog.text
    assert len(config.filepaths) == (len(files_df) - 2)

    config = TrainConfig(labels=tmp_path / "labels_with_non_loadable_videos.csv")
    assert "Skipping 2 file(s) that could not be loaded with ffmpeg" in caplog.text
    assert len(config.labels) == (len(files_df) - 2)


def test_empty_model_config():
    with pytest.raises(ValueError) as error:
        ModelConfig()
    assert (
        "Must provide either `train_config` or `predict_config`" in error.value.errors()[0]["msg"]
    )


def test_early_stopping_mode():
    assert EarlyStoppingConfig(monitor="val_macro_f1").mode == "max"
    assert EarlyStoppingConfig(monitor="val_loss").mode == "min"

    with pytest.raises(ValidationError) as error:
        # if you really want to do the wrong thing you have to be explicit about it
        EarlyStoppingConfig(monitor="val_loss", mode="max")

    assert "Provided mode max is incorrect for val_loss monitor." == error.value.errors()[0]["msg"]


def test_labels_with_all_null_species(labels_absolute_path):
    labels = pd.read_csv(labels_absolute_path)
    labels["label"] = np.nan
    with pytest.raises(ValueError) as error:
        TrainConfig(labels=labels)
    assert "Species cannot be null for all videos." == error.value.errors()[0]["msg"]


def test_labels_with_partially_null_species(labels_absolute_path, caplog):
    labels = pd.read_csv(labels_absolute_path)
    labels.loc[0, "label"] = np.nan
    TrainConfig(labels=labels)
    assert "Found 1 filepath(s) with no label. Will skip." in caplog.text


def test_labels_with_all_null_split(labels_absolute_path, caplog):
    labels = pd.read_csv(labels_absolute_path)
    labels["split"] = np.nan
    TrainConfig(labels=labels)
    assert "Split column is entirely null. Will generate splits automatically" in caplog.text


def test_labels_with_partially_null_split(labels_absolute_path):
    labels = pd.read_csv(labels_absolute_path)
    labels.loc[0, "split"] = np.nan
    with pytest.raises(ValueError) as error:
        TrainConfig(labels=labels)
    assert (
        "Found 1 row(s) with null `split`. Fill in these rows with either `train`, `val`, or `holdout`"
    ) in error.value.errors()[0]["msg"]


def test_labels_with_invalid_split(labels_absolute_path):
    labels = pd.read_csv(labels_absolute_path)
    labels.loc[0, "split"] = "test"
    with pytest.raises(ValueError) as error:
        TrainConfig(labels=labels)
    assert (
        "Found the following invalid values for `split`: {'test'}. `split` can only contain `train`, `val`, or `holdout.`"
    ) == error.value.errors()[0]["msg"]


def test_labels_no_splits(labels_no_splits, tmp_path):
    config = TrainConfig(data_dir=TEST_VIDEOS_DIR, labels=labels_no_splits, save_dir=tmp_path)
    assert set(config.labels.split.unique()) == set(("holdout", "train", "val"))


def test_labels_split_proportions(labels_no_splits, tmp_path):
    config = TrainConfig(
        data_dir=TEST_VIDEOS_DIR,
        labels=labels_no_splits,
        split_proportions={"a": 3, "b": 1},
        save_dir=tmp_path,
    )
    assert config.labels.split.value_counts().to_dict() == {"a": 14, "b": 5}


def test_from_scratch(labels_absolute_path):
    config = TrainConfig(labels=labels_absolute_path, from_scratch=True, checkpoint=None)
    assert config.model_name == "time_distributed"
    assert config.from_scratch
    assert config.checkpoint is None

    with pytest.raises(ValueError) as error:
        TrainConfig(labels=labels_absolute_path, from_scratch=True, model_name=None)
    assert "If from_scratch=True, model_name cannot be None." == error.value.errors()[0]["msg"]


def test_predict_dry_run_and_save(labels_absolute_path, caplog, tmp_path):
    config = PredictConfig(filepaths=labels_absolute_path, dry_run=True, save=True)
    assert (
        "Cannot save when predicting with dry_run=True. Setting save=False and save_dir=None."
        in caplog.text
    )
    assert not config.save
    assert config.save_dir is None

    config = PredictConfig(filepaths=labels_absolute_path, dry_run=True, save_dir=tmp_path)
    assert not config.save
    assert config.save_dir is None


def test_predict_filepaths_with_duplicates(labels_absolute_path, tmp_path, caplog):
    filepaths = pd.read_csv(labels_absolute_path, usecols=["filepath"])
    # add duplicate filepath
    filepaths.append(filepaths.loc[0]).to_csv(tmp_path / "filepaths_with_dupe.csv")

    PredictConfig(filepaths=tmp_path / "filepaths_with_dupe.csv")
    assert "Found 1 duplicate row(s) in filepaths csv. Dropping duplicates" in caplog.text


def test_model_cache_dir(labels_absolute_path, tmp_path):
    config = TrainConfig(labels=labels_absolute_path)
    assert config.model_cache_dir == Path(appdirs.user_cache_dir()) / "zamba"

    os.environ["MODEL_CACHE_DIR"] = str(tmp_path)
    config = TrainConfig(labels=labels_absolute_path)
    assert config.model_cache_dir == tmp_path

    config = PredictConfig(filepaths=labels_absolute_path, model_cache_dir=tmp_path / "my_cache")
    assert config.model_cache_dir == tmp_path / "my_cache"


def test_predict_save(labels_absolute_path, tmp_path, dummy_trained_model_checkpoint):
    # if save is True, save in current working directory
    config = PredictConfig(filepaths=labels_absolute_path, skip_load_validation=True)
    assert config.save_dir == Path.cwd()

    config = PredictConfig(filepaths=labels_absolute_path, save=False, skip_load_validation=True)
    assert config.save is False
    assert config.save_dir is None

    # if save_dir is specified, set save to True
    config = PredictConfig(
        filepaths=labels_absolute_path,
        save=False,
        save_dir=tmp_path / "my_dir",
        skip_load_validation=True,
    )
    assert config.save is True
    # save dir gets created
    assert (tmp_path / "my_dir").exists()

    # empty save dir does not error
    save_dir = tmp_path / "save_dir"
    save_dir.mkdir()

    config = PredictConfig(
        filepaths=labels_absolute_path, save_dir=save_dir, skip_load_validation=True
    )
    assert config.save_dir == save_dir

    # save dir with prediction csv or yaml will error
    for pred_file in [
        (save_dir / "zamba_predictions.csv"),
        (save_dir / "predict_configuration.yaml"),
    ]:
        # just takes one of the two files to raise error
        pred_file.touch()
        with pytest.raises(ValueError) as error:
            PredictConfig(
                filepaths=labels_absolute_path, save_dir=save_dir, skip_load_validation=True
            )
        assert (
            f"zamba_predictions.csv and/or predict_configuration.yaml already exist in {save_dir}. If you would like to overwrite, set overwrite=True"
            == error.value.errors()[0]["msg"]
        )
        pred_file.unlink()

    # can overwrite
    pred_file.touch()
    config = PredictConfig(
        filepaths=labels_absolute_path,
        save_dir=save_dir,
        skip_load_validation=True,
        overwrite=True,
    )
    assert config.save_dir == save_dir


def test_validate_scheduler(labels_absolute_path):
    # None gets transformed into SchedulerConfig
    config = TrainConfig(
        labels=labels_absolute_path, scheduler_config=None, skip_load_validation=True
    )
    assert config.scheduler_config == SchedulerConfig(scheduler=None, scheduler_params=None)

    # default is valid
    config = TrainConfig(
        labels=labels_absolute_path, scheduler_config="default", skip_load_validation=True
    )
    assert config.scheduler_config == "default"

    # other strings are not
    with pytest.raises(ValueError) as error:
        TrainConfig(
            labels=labels_absolute_path, scheduler_config="StepLR", skip_load_validation=True
        )
    assert (
        "Scheduler can either be 'default', None, or a SchedulerConfig."
        == error.value.errors()[0]["msg"]
    )

    # custom scheduler
    config = TrainConfig(
        labels=labels_absolute_path,
        scheduler_config=SchedulerConfig(scheduler="StepLR", scheduler_params={"gamma": 0.2}),
        skip_load_validation=True,
    )
    assert config.scheduler_config == SchedulerConfig(
        scheduler="StepLR", scheduler_params={"gamma": 0.2}
    )


def test_dry_run_and_skip_load_validation(labels_absolute_path, caplog):
    # check dry_run is True sets skip_load_validation to True
    config = TrainConfig(labels=labels_absolute_path, dry_run=True, skip_load_validation=False)
    assert config.skip_load_validation
    assert "Turning off video loading check since dry_run=True." in caplog.text

    # if dry run is False, skip_load_validation is unchanged
    config = TrainConfig(labels=labels_absolute_path, dry_run=False, skip_load_validation=False)
    assert not config.skip_load_validation


def test_default_video_loader_config(labels_absolute_path):
    # if no video loader is specified, use default for model
    config = ModelConfig(
        train_config=TrainConfig(labels=labels_absolute_path, skip_load_validation=True),
        video_loader_config=None,
    )
    assert config.video_loader_config is not None

    config = ModelConfig(
        predict_config=PredictConfig(filepaths=labels_absolute_path, skip_load_validation=True),
        video_loader_config=None,
    )
    assert config.video_loader_config is not None


def test_checkpoint_sets_model_to_none(labels_absolute_path, dummy_trained_model_checkpoint):
    config = TrainConfig(
        labels=labels_absolute_path,
        checkpoint=dummy_trained_model_checkpoint,
        skip_load_validation=True,
    )
    assert config.model_name is None
