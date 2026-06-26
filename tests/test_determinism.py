import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from zamba.object_detection.yolox.megadetector_lite_yolox import _rank_frame_indices
from zamba.pytorch.utils import configure_inference_determinism

pytestmark = pytest.mark.video


def test_configure_inference_determinism_seeds_and_sets_cublas(monkeypatch):
    seed_calls = []
    monkeypatch.setattr(
        "zamba.pytorch.utils.pl.seed_everything",
        lambda seed, workers=False: seed_calls.append((seed, workers)),
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    configure_inference_determinism(deterministic=True)

    assert seed_calls == [(55, True)]
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_configure_inference_determinism_custom_seed(monkeypatch):
    seed_calls = []
    monkeypatch.setattr(
        "zamba.pytorch.utils.pl.seed_everything",
        lambda seed, workers=False: seed_calls.append((seed, workers)),
    )

    configure_inference_determinism(seed=123, deterministic=True)

    assert seed_calls == [(123, True)]


def test_configure_inference_determinism_skips_cublas_when_disabled(monkeypatch):
    seed_calls = []
    monkeypatch.setattr(
        "zamba.pytorch.utils.pl.seed_everything",
        lambda seed, workers=False: seed_calls.append((seed, workers)),
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    configure_inference_determinism(deterministic=False)

    assert seed_calls == [(55, True)]
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_configure_inference_determinism_preserves_existing_cublas_config(monkeypatch):
    monkeypatch.setattr("zamba.pytorch.utils.pl.seed_everything", MagicMock())
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    configure_inference_determinism(deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_rank_frame_indices_is_deterministic():
    scores = np.array([1.0, 1.0, 0.5, 1.0, 0.0])
    first = _rank_frame_indices(scores)
    second = _rank_frame_indices(scores)
    np.testing.assert_array_equal(first, second)
    assert list(first) == [0, 1, 3, 2, 4]
