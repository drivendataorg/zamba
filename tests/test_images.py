import json
import logging
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from zamba.images.bbox import BboxInputFormat, bbox_json_to_df, BboxLayout
from zamba.images.classifier import ImageClassifierModule
from zamba.images.config import ImageClassificationTrainingConfig
from zamba.images.data import absolute_bbox
from zamba.images.dataset.dataset import crop_image, prepare_dataset
from zamba.images.manager import train
from zamba.images.result import results_to_megadetector_format

from conftest import ASSETS_DIR, DummyZambaImageClassificationLightningModule


@pytest.fixture
def annotations():
    with open(ASSETS_DIR / "labels.json", "r") as f:
        return json.load(f)


@pytest.fixture
def annotation():
    return {
        "id": "0003",
        "category_id": 6,
        "image_id": "ffe3",
        "bbox": [131, 220, 178, 319],
    }


@pytest.fixture
def dog():
    return Image.open(ASSETS_DIR / "dog.jpg")


@pytest.fixture
def trimmed_dog():
    return Image.open(ASSETS_DIR / "images/crop/dog.jpg")


@pytest.fixture
def labels_path():
    return ASSETS_DIR / "images/labels.csv"


@pytest.fixture
def images_path():
    return ASSETS_DIR


@pytest.fixture
def megadetector_output_path():
    return ASSETS_DIR / "images/megadetector_output.json"


@pytest.fixture
def dataframe_result_csv_path():
    return ASSETS_DIR / "images/df_result.csv"


@pytest.fixture
def splits():
    return {"splits": {"train": ["A", "B"], "test": ["C"], "val": ["D"]}}


@pytest.fixture
def categories_map():
    return {
        "Test1": "Test1",
        "Test2": "Test2",
        "Test3": "Test3",
        "Test4": "Test4",
        "Test5": "Test5",
        "Test6": "Test6",
    }


@pytest.fixture
def dummy_checkpoint(labels_path, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dummy-model-dir")
    labels = pd.read_csv(labels_path)
    species = list(labels.label.unique())
    output_path = tmp_path / "dummy.ckpt"
    DummyZambaImageClassificationLightningModule(num_hidden=1, species=species).to_disk(
        output_path
    )
    return output_path


def test_prepare_dataset_split(annotations, categories_map, splits):
    result = prepare_dataset(
        annotations,
        splits,
        "s3://example.example/example",
        categories_map,
        "example-name",
    )

    assert len(result) == 4
    assert len(result[result["split"] == "train"]) == 2
    assert len(result[result["split"] == "test"]) == 1
    assert len(result[result["split"] == "val"]) == 1


def test_image_annotation_crop_images_the_same_size(annotation, dog, trimmed_dog):
    result = crop_image(dog, annotation["bbox"])

    assert result.height == trimmed_dog.height
    assert result.width == trimmed_dog.width


def test_train_config_validate_labels_from_path(labels_path, images_path):
    config = ImageClassificationTrainingConfig(data_dir=images_path, labels=labels_path)

    assert isinstance(config.labels, pd.DataFrame)


def test_train_config_labels(labels_path, images_path):
    config = ImageClassificationTrainingConfig(data_dir=images_path, labels=labels_path)
    logging.warning(config.labels.head())
    assert "label" in config.labels.columns


def test_train_config_data_exist(labels_path, images_path):
    config = ImageClassificationTrainingConfig(data_dir=images_path, labels=labels_path)

    assert len(config.labels) == 5


def test_absolute_bbox():
    image = Image.fromarray(np.zeros((600, 800)))
    bbox = [0.1, 0.25, 0.25, 0.5]

    result = absolute_bbox(image, bbox, bbox_layout=BboxLayout.XYWH)

    assert result[0] == 80
    assert result[1] == 150
    assert result[2] == 280
    assert result[3] == 450
    assert len(result) == 4


def test_absolute_bbox_xy():
    image = Image.fromarray(np.zeros((600, 800)))
    bbox = [200, 200, 400, 400]

    result = absolute_bbox(image, bbox, bbox_layout=BboxLayout.XYXY)

    assert result[0] == 200
    assert result[1] == 200
    assert result[2] == 400
    assert result[3] == 400
    assert len(result) == 4


def test_image_crop_from_megadetector_bbox(dog, trimmed_dog):
    bbox = absolute_bbox(dog, [0.1718, 0.3836, 0.233, 0.5555], bbox_layout=BboxLayout.XYWH)

    result = dog.crop(bbox)
    logging.warning(result)

    assert result.height == trimmed_dog.height
    assert result.width == trimmed_dog.width


@pytest.mark.parametrize("model_class", (ImageClassifierModule,))
def test_save_and_load(model_class, tmp_path):
    model = model_class(
        species=["cat", "dog"], batch_size=2, image_size=224, model_name="resnet50"
    )
    model.to_disk(tmp_path / model_class.__name__)
    model = model_class.from_disk(tmp_path / model_class.__name__)
    assert model.species == ["cat", "dog"]
    assert model.num_classes == 2


def test_bbox_json_to_df_format_megadetector(megadetector_output_path):
    with open(megadetector_output_path, "r") as f:
        bbox_json = json.load(f)

    result = bbox_json_to_df(bbox_json, BboxInputFormat.MEGADETECTOR)

    assert len(result) == 2
    assert all(k in result.columns for k in ["x1", "x2", "y1", "y2"])

    assert result.iloc[0]["filepath"] == "path/to/example.jpg"
    assert result.iloc[0]["label"] == "bear"
    assert result.iloc[0]["label_id"] == "3"

    assert result.iloc[1]["label"] == "wolf"
    assert result.iloc[1]["label_id"] == "2"


def test_results_to_megadetector_format(dataframe_result_csv_path):
    df = pd.read_csv(dataframe_result_csv_path)
    species = df.filter(like="species_").columns.tolist()

    result = results_to_megadetector_format(df, species)

    assert len(result.images) == 2
    assert len(result.images[0].detections) == 2
    assert len(result.images[1].detections) == 1


def test_train_integration(images_path, labels_path, dummy_checkpoint, tmp_path):
    save_dir = tmp_path / "my_model"
    checkpoint_path = tmp_path / "checkpoints"

    config = ImageClassificationTrainingConfig(
        data_dir=images_path,
        labels=labels_path,
        model_name="dummy",
        max_epochs=1,
        batch_size=1,
        checkpoint=dummy_checkpoint,
        checkpoint_path=checkpoint_path,
        from_scratch=False,
        save_dir=save_dir,
        # force CPU since MPS is unsupported on macOS github actions runners
        accelerator="cpu" if os.getenv("RUNNER_OS") == "macOS" else None,
    )

    train(config)
    assert save_dir.exists()

    for f in ["train_configuration.yaml", "val_metrics.json", f"{config.model_name}.ckpt"]:
        assert (config.save_dir / f).exists()
