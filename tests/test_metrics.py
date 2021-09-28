import numpy as np
import pytest
from zamba.metrics import compute_species_specific_metrics


@pytest.fixture
def y_true():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Mutually exclusive labels
    return np.c_[1 - y_true, y_true]


@pytest.fixture
def y_pred():
    y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 0])
    # Mutually exclusive predictions
    return np.c_[1 - y_pred, y_pred]


def test_compute_species_specific_metrics(y_true, y_pred):
    metrics = {
        f"{metric_name}/{index}": metric
        for metric_name, index, metric in compute_species_specific_metrics(y_true, y_pred)
    }

    assert metrics == {
        "accuracy/0": 0.5,
        "f1/0": 0.3333333333333333,
        "precision/0": 0.5,  # Label `0` predicted twice, only once was actually `0`
        "recall/0": 0.25,  # Of four `0` labels, only one was predicted correctly
        "accuracy/1": 0.5,
        "f1/1": 0.6,
        "precision/1": 0.5,  # Label `1` predicted six times, only three were actually `1`
        "recall/1": 0.75,  # Of four `1` labels, only three were predicted correctly
    }


def test_compute_species_specific_metrics_with_labels(y_true, y_pred):
    metrics = {
        f"{metric_name}/{index}": metric
        for metric_name, index, metric in compute_species_specific_metrics(
            y_true, y_pred, labels=["frog", "caterpillar"]
        )
    }

    assert metrics == {
        "accuracy/frog": 0.5,
        "f1/frog": 0.3333333333333333,
        "precision/frog": 0.5,  # Label `0` predicted twice, only once was actually `0`
        "recall/frog": 0.25,  # Of four `0` labels, only one was predicted correctly
        "accuracy/caterpillar": 0.5,
        "f1/caterpillar": 0.6,
        "precision/caterpillar": 0.5,  # Label `1` predicted six times, only three were actually `1`
        "recall/caterpillar": 0.75,  # Of four `1` labels, only three were predicted correctly
    }


def test_compute_species_specific_metrics_wrong_number_of_labels(y_true, y_pred):
    with pytest.raises(ValueError) as error:
        list(
            compute_species_specific_metrics(
                y_true, y_pred, labels=["frog", "caterpillar", "squid"]
            )
        )
    assert (
        "The number of labels (3) must match the number of columns in y_true (2)."
        == error.value.args[0]
    )
