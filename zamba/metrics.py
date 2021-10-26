from typing import Generator, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_species_specific_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Generator[Tuple[str, int, float], None, None]:
    """Computes species-specific accuracy, F1, precision, and recall.
    Args:
        y_true (np.ndarray): An array with shape (samples, species) where each value indicates
            the presence of a species in a sample.
        y_pred (np.ndarray): An array with shape (samples, species) where each value indicates
            the predicted presence of a species in a sample.

    Yields:
        str, int, float: The metric name, species label index, and metric value.
    """
    if labels is None:
        labels = range(y_true.shape[1])

    elif len(labels) != y_true.shape[1]:
        raise ValueError(
            f"The number of labels ({len(labels)}) must match the number of columns in y_true ({y_true.shape[1]})."
        )

    for index, label in enumerate(labels):
        yield "accuracy", label, accuracy_score(y_true[:, index], y_pred[:, index])
        yield "f1", label, f1_score(y_true[:, index], y_pred[:, index], zero_division=0)
        yield "precision", label, precision_score(
            y_true[:, index], y_pred[:, index], zero_division=0
        )
        yield "recall", label, recall_score(y_true[:, index], y_pred[:, index], zero_division=0)
