"""Tests for the class-balanced training sampler (config.balanced_sampling)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torchvision.transforms import transforms

from zamba.images.data import ImageClassificationDataModule
from zamba.images.manager import get_train_augmentations


def _imbalanced_annotations(tmp_path):
    """Two common classes (100 each) and one rare class (5), already integer-encoded
    the way ImageClassificationTrainingConfig.preprocess_labels produces them."""
    return pd.DataFrame(
        {
            "filepath": [f"img{i}.jpg" for i in range(205)],
            "label": [0] * 100 + [1] * 100 + [2] * 5,
            "split": ["train"] * 205,
        }
    )


def _datamodule(tmp_path, balanced):
    return ImageClassificationDataModule(
        data_dir=tmp_path,
        annotations=_imbalanced_annotations(tmp_path),
        cache_dir=tmp_path,
        crop_images=False,          # skip MegaDetector/crop preprocessing
        batch_size=16,
        num_workers=0,
        balanced_sampling=balanced,
    )


def test_default_is_unchanged(tmp_path):
    """balanced_sampling=False keeps the original shuffle behavior (RandomSampler)."""
    dl = _datamodule(tmp_path, balanced=False).train_dataloader()
    assert isinstance(dl.sampler, RandomSampler)


def test_balanced_sampling_uses_weighted_sampler(tmp_path):
    dl = _datamodule(tmp_path, balanced=True).train_dataloader()
    assert isinstance(dl.sampler, WeightedRandomSampler)
    # one epoch still sees as many images as there are training rows
    assert dl.sampler.num_samples == 205


def test_balanced_sampling_equalizes_classes(tmp_path):
    """Over a full epoch the rare class should be drawn on the order of as often as
    each common class, rather than at its 5/205 natural rate."""
    import torch

    torch.manual_seed(0)
    dm = _datamodule(tmp_path, balanced=True)
    labels = dm.annotations["label"].to_numpy()
    drawn = labels[np.fromiter(iter(dm.train_dataloader().sampler), dtype=int)]
    seen = np.bincount(drawn, minlength=3)
    # every class within 40% of the mean draw count (natural rate would be ~2%)
    assert (np.abs(seen - seen.mean()) / seen.mean() < 0.4).all()


def test_no_zero_division_when_class_absent_from_train(tmp_path):
    """A class present in the label space but absent from the train split must not
    produce inf/nan weights (weights index only classes that actually appear)."""
    ann = _imbalanced_annotations(tmp_path)
    ann.loc[ann["label"] == 2, "split"] = "val"  # class 2 no longer in train
    dm = ImageClassificationDataModule(
        data_dir=tmp_path, annotations=ann, cache_dir=tmp_path,
        crop_images=False, batch_size=16, num_workers=0, balanced_sampling=True,
    )
    weights = np.asarray(dm.train_dataloader().sampler.weights)
    assert np.isfinite(weights).all()


def _augment_config(balanced, extra=False, image_size=224):
    """get_train_augmentations only reads these three attributes off the config."""
    return SimpleNamespace(
        image_size=image_size,
        balanced_sampling=balanced,
        extra_train_augmentations=extra,
    )


def test_balanced_sampling_enables_strong_augmentations():
    """Turning on balanced_sampling automatically brings the heavier augmentation stack:
    photometric jitter + grayscale + blur (for diversity) plus RandomErasing occlusion."""
    augment, final = get_train_augmentations(_augment_config(balanced=True))
    types = {type(t) for t in augment}
    # the diversity-generating photometric transforms the mild default lacks
    assert transforms.ColorJitter in types
    assert transforms.RandomGrayscale in types
    assert transforms.RandomResizedCrop in types
    # RandomErasing is applied last, on the normalized tensor
    assert len(final) == 1 and isinstance(final[0], transforms.RandomErasing)


def test_default_augmentations_are_mild():
    """Without balanced_sampling or extra_train_augmentations, no photometric/erasing stack."""
    augment, final = get_train_augmentations(_augment_config(balanced=False))
    types = {type(t) for t in augment}
    assert transforms.ColorJitter not in types
    assert transforms.RandomGrayscale not in types
    assert final == []


def test_balanced_sampling_augmentations_take_precedence_over_extra():
    """When both flags are set, the (heavier) balanced-sampling stack wins."""
    both = get_train_augmentations(_augment_config(balanced=True, extra=True))
    only_balanced = get_train_augmentations(_augment_config(balanced=True, extra=False))
    assert [type(t) for t in both[0]] == [type(t) for t in only_balanced[0]]
    # the balanced stack has ColorJitter, which the extra_train_augmentations stack does not
    assert any(isinstance(t, transforms.ColorJitter) for t in both[0])


def test_balanced_augmentations_respect_image_size():
    """The RandomResizedCrop target size follows the resolved training image size."""
    augment, _ = get_train_augmentations(_augment_config(balanced=True, image_size=480))
    crop = next(t for t in augment if isinstance(t, transforms.RandomResizedCrop))
    assert tuple(crop.size) == (480, 480)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
