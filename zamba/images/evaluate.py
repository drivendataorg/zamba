import io
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from zamba.metrics import compute_species_specific_metrics


class ClassificationEvaluator:
    def __init__(
        self,
        labels: List[str],
    ):
        self.labels = labels
        self.num_classes = len(self.labels)

    def get_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {**self.top_k_accuracy_data(y_true, y_pred), **self.overall_metrics(y_true, y_pred)}

    def species_score_metrics(self, y_true, y_pred) -> Dict[str, float]:
        species_metrics = {}

        for metric, label, value in compute_species_specific_metrics(y_true, y_pred, self.labels):
            species_metrics[f"{metric}_{label}"] = value

        return species_metrics

    def confusion_matrix_data(self, y_true, y_pred):
        return confusion_matrix(
            y_true.argmax(axis=1),
            y_pred.argmax(axis=1),
            labels=[i for i in range(self.num_classes)],
            normalize="true",
        )

    def confusion_matrix_plot(self, y_true, y_pred) -> Optional[Image]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        cm = self.confusion_matrix_data(y_true, y_pred)

        size = min(max(12, int(1.5 * self.num_classes)), 30)
        fig, ax = plt.subplots(figsize=(size, size), dpi=150)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        plt.colorbar(cax)

        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_xticklabels(self.labels)
        ax.set_yticklabels(self.labels)

        plt.xticks(rotation=45, ha="left")

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Normalized Confusion Matrix")

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)

        return Image.open(buf)

    @staticmethod
    def overall_metrics(y_true, y_pred) -> Dict[str, float]:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(y_true, axis=1)

        return {
            "accuracy": accuracy_score(y_test_labels, y_pred_labels),  # zero_division
            "recall": recall_score(
                y_test_labels, y_pred_labels, average="macro", zero_division=False
            ),
            "precision": precision_score(
                y_test_labels, y_pred_labels, average="macro", zero_division=False
            ),
            "f1": f1_score(y_test_labels, y_pred_labels, average="macro", zero_division=False),
            "weighted_recall": recall_score(
                y_test_labels, y_pred_labels, average="weighted", zero_division=False
            ),
            "weighted_precision": precision_score(
                y_test_labels, y_pred_labels, average="weighted", zero_division=False
            ),
            "weighted_f1": f1_score(
                y_test_labels, y_pred_labels, average="weighted", zero_division=False
            ),
        }

    def top_k_accuracy_data(self, y_true, y_pred, ks: Optional[List[int]] = None):
        if ks is None:
            ks = [1, 3, 5, 10]
        k_scores = {
            f"top_{k}_accuracy": top_k_accuracy_score(
                y_true.argmax(axis=1), y_pred, k=k, labels=list(range(self.num_classes))
            )
            for k in ks
        }

        return k_scores
