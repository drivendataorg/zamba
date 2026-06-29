import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.image

from zamba.images.bbox import BboxInputFormat, bbox_json_to_df, BboxLayout  # noqa: E402
from zamba.images.classifier import ImageClassifierModule, infer_model_family  # noqa: E402
from zamba.images.config import (  # noqa: E402
    ImageClassificationPredictConfig,
    ImageClassificationTrainingConfig,
)
from zamba.images.data import absolute_bbox  # noqa: E402
from zamba.images.dataset.dataset import crop_image, prepare_dataset  # noqa: E402
from zamba.images.manager import (  # noqa: E402
    get_default_transforms,
    resolve_inference_family,
    resolve_training_image_size,
    train,
)
from zamba.images.result import results_to_megadetector_format  # noqa: E402

from conftest import ASSETS_DIR, DummyZambaImageClassificationLightningModule  # noqa: E402


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


def test_train_config_debug_sampling(labels_path, images_path):
    full = ImageClassificationTrainingConfig(data_dir=images_path, labels=labels_path)
    debug = ImageClassificationTrainingConfig(data_dir=images_path, labels=labels_path, debug=1)

    assert len(debug.labels) <= len(full.labels)
    # debug caps the data at N images per (split, class)
    assert (debug.labels.groupby(["split", "label"]).size() <= 1).all()


def test_train_config_debug_must_be_positive(labels_path, images_path):
    with pytest.raises(ValueError, match="debug must be a positive integer"):
        ImageClassificationTrainingConfig(data_dir=images_path, labels=labels_path, debug=0)


def test_train_config_debug_with_no_matching_split_rows(labels_path, images_path):
    """If the provided splits don't include train/val/test, debug sampling finds nothing
    to sample and leaves the labels untouched (rather than erroring)."""
    labels = pd.read_csv(labels_path)
    labels["split"] = "holdout"  # none of train/val/test

    config = ImageClassificationTrainingConfig(data_dir=images_path, labels=labels, debug=1)
    # nothing was sampled, so all (existing) rows are retained
    assert (config.labels["split"] == "holdout").all()


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Keep predict-config tests off the repo root (save paths default to cwd)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_predict_config_rejects_nonpositive_image_size(images_path, isolated_cwd):
    with pytest.raises(ValueError, match="Image size should be greater than 0"):
        ImageClassificationPredictConfig(
            data_dir=images_path,
            filepaths=pd.DataFrame({"filepath": ["dog.jpg"]}),
            image_size=0,
            save=False,
        )


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


@pytest.mark.parametrize(
    "name,expected",
    [
        ("tf_efficientnetv2_m", "speciesnet"),
        ("speciesnet", "speciesnet"),
        ("convnextv2_base.fcmae_ft_in22k_in1k", "lila.science"),
        ("resnet50", "lila.science"),
        (None, "lila.science"),
    ],
)
def test_infer_model_family(name, expected):
    assert infer_model_family(name) == expected


def test_get_default_transforms_speciesnet_has_no_normalization():
    """SpeciesNet uses 480px bicubic resize and NO normalization, unlike lila.science."""
    from torchvision.transforms import transforms

    top_s, bottom_s, size_s = get_default_transforms("speciesnet", None)
    assert size_s == 480
    assert isinstance(top_s[0], transforms.Resize)
    assert not any(isinstance(t, transforms.Normalize) for t in bottom_s)

    top_l, bottom_l, size_l = get_default_transforms("lila.science", None)
    assert size_l == 224
    assert any(isinstance(t, transforms.Normalize) for t in bottom_l)

    # an unknown family behaves like the generic (lila.science) pipeline
    _, bottom_g, _ = get_default_transforms("something_else", None)
    assert any(isinstance(t, transforms.Normalize) for t in bottom_g)

    # tuple image_size (as stored on speciesnet checkpoints) is normalized to int
    _, _, size_t = get_default_transforms("speciesnet", (480, 480))
    assert size_t == 480


def test_image_classifier_persists_and_resolves_model_family(tmp_path):
    """model_family round-trips through a checkpoint and is inferred for legacy ones."""
    model = ImageClassifierModule(
        species=["cat", "dog"], batch_size=2, image_size=224, model_name="resnet50"
    )
    # inferred from arch name; persisted to hparams so prediction can read it back
    assert model.model_family == "lila.science"
    assert model.hparams["model_family"] == "lila.science"

    path = tmp_path / "m.ckpt"
    model.to_disk(path)
    reloaded = ImageClassifierModule.from_disk(path)
    assert reloaded.model_family == "lila.science"


def test_model_family_explicit_and_legacy_zamba_model_override():
    explicit = ImageClassifierModule(
        species=["cat"],
        batch_size=1,
        image_size=480,
        model_name="resnet50",
        model_family="speciesnet",
    )
    assert explicit.model_family == "speciesnet"

    # converted SpeciesNet weights carry the family under the legacy `zamba_model` key
    legacy = ImageClassifierModule(
        species=["cat"],
        batch_size=1,
        image_size=480,
        model_name="resnet50",
        zamba_model="speciesnet",
    )
    assert legacy.model_family == "speciesnet"


def test_resolve_inference_family_prefers_checkpoint(tmp_path):
    """A provided checkpoint is authoritative over a (possibly stale) model_name."""
    model = ImageClassifierModule(
        species=["cat", "dog"],
        batch_size=2,
        image_size=480,
        model_name="resnet50",
        model_family="speciesnet",
    )
    path = tmp_path / "ck.ckpt"
    model.to_disk(path)

    # model_name defaults to lila.science, but the checkpoint says speciesnet -> wins
    assert resolve_inference_family("lila.science", path) == "speciesnet"
    # with no checkpoint, fall back to inference from model_name
    assert resolve_inference_family("lila.science", None) == "lila.science"


def test_resolve_inference_family_falls_back_on_unreadable_checkpoint(tmp_path):
    """A corrupt/unreadable checkpoint must not crash prediction; fall back to model_name."""
    bad_ckpt = tmp_path / "corrupt.ckpt"
    bad_ckpt.write_text("not a real checkpoint")
    assert resolve_inference_family("speciesnet", bad_ckpt) == "speciesnet"


def test_predict_config_resolves_model_name_from_checkpoint_family(isolated_cwd, images_path):
    """An image checkpoint has no _default_model_name, so model_name falls back to the
    persisted preprocessing family rather than the (conflicting) passed model_name."""
    model = ImageClassifierModule(
        species=["cat", "dog"],
        batch_size=1,
        image_size=480,
        model_name="resnet50",
        model_family="speciesnet",
    )
    ckpt = isolated_cwd / "ck.ckpt"
    model.to_disk(ckpt)

    config = ImageClassificationPredictConfig(
        data_dir=images_path,
        filepaths=pd.DataFrame({"filepath": ["dog.jpg"]}),
        checkpoint=ckpt,
        model_name="lila.science",
        save=False,
    )
    assert config.model_name == "speciesnet"


@pytest.mark.parametrize(
    "model_name,head_path",
    [
        ("convnext_atto", ("head", "fc")),  # head.fc layout
        ("efficientnet_b0", ("classifier",)),  # classifier layout
    ],
)
def test_finetune_from_replaces_head(model_name, head_path, tmp_path):
    """Finetuning from a checkpoint rebuilds the classification head to the new class
    count, handling both the `head.fc` (convnext) and `classifier` (efficientnet) layouts."""
    base = ImageClassifierModule(
        species=["cat", "dog"], batch_size=1, image_size=224, model_name=model_name
    )
    ckpt = tmp_path / f"{model_name}.ckpt"
    base.to_disk(ckpt)

    finetuned = ImageClassifierModule(
        species=["cat", "dog", "bird"],
        batch_size=1,
        image_size=224,
        model_name=model_name,
        finetune_from=ckpt,
    )
    assert finetuned.num_classes == 3

    head = finetuned.model
    for attr in head_path:
        head = getattr(head, attr)
    assert head.out_features == 3


def test_resolve_training_image_size_prefers_checkpoint(tmp_path):
    """Without an explicit image size, the checkpoint's own image size wins over the
    family default; an explicit size always wins, and training from scratch ignores it."""
    ckpt = tmp_path / "ck.ckpt"
    torch.save({"hyper_parameters": {"image_size": (384, 384)}}, ckpt)

    # no explicit size + checkpoint -> use the checkpoint's size
    no_explicit = SimpleNamespace(image_size=None, checkpoint=ckpt, from_scratch=False)
    assert resolve_training_image_size(no_explicit) == (384, 384)

    # explicit size always wins
    explicit = SimpleNamespace(image_size=256, checkpoint=ckpt, from_scratch=False)
    assert resolve_training_image_size(explicit) == 256

    # training from scratch ignores the checkpoint and defers to the family default
    scratch = SimpleNamespace(image_size=None, checkpoint=ckpt, from_scratch=True)
    assert resolve_training_image_size(scratch) is None

    # no checkpoint -> defer to the family default
    none_ckpt = SimpleNamespace(image_size=None, checkpoint=None, from_scratch=False)
    assert resolve_training_image_size(none_ckpt) is None


def test_on_load_checkpoint_remaps_unprefixed_speciesnet_weights():
    """SpeciesNet weights are saved without the lightning 'model.' prefix; on_load_checkpoint
    must add it for keys that belong to the model subtree and leave everything else alone."""
    model = DummyZambaImageClassificationLightningModule(species=["a", "b"])
    expected = model.state_dict()

    # speciesnet-style: model weights present but WITHOUT the "model." prefix, plus an
    # unrelated key that must be passed through untouched.
    unprefixed = {k[len("model.") :]: v for k, v in expected.items() if k.startswith("model.")}
    unprefixed["unrelated_buffer"] = torch.zeros(1)

    ckpt = {"state_dict": dict(unprefixed)}
    model.on_load_checkpoint(ckpt)
    remapped = ckpt["state_dict"]
    assert all(k.startswith("model.") for k in remapped if k != "unrelated_buffer")
    assert "unrelated_buffer" in remapped

    # already-prefixed (normal zamba) checkpoints are left unchanged
    normal = {"state_dict": dict(expected)}
    model.on_load_checkpoint(normal)
    assert set(normal["state_dict"]) == set(expected)

    # empty state dict is a no-op
    empty = {"state_dict": {}}
    model.on_load_checkpoint(empty)
    assert empty["state_dict"] == {}


def test_resize_and_pad_accepts_scalar_and_tuple(dog):
    from zamba.pytorch.transforms import resize_and_pad

    assert resize_and_pad(dog, desired_size=64).size == (64, 64)
    assert resize_and_pad(dog, desired_size=(64, 32)).size == (64, 32)


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


@pytest.mark.parametrize("extra_train_augmentations", [False, True])
def test_train_integration(
    extra_train_augmentations, images_path, labels_path, dummy_checkpoint, tmp_path
):
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
        extra_train_augmentations=extra_train_augmentations,
    )

    train(config)
    assert save_dir.exists()

    for f in ["train_configuration.yaml", "val_metrics.json", f"{config.model_name}.ckpt"]:
        assert (config.save_dir / f).exists()
