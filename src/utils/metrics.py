"""
Metrics Utilities
Helpers for computing and formatting evaluation metrics.
"""

from typing import List, Dict
import numpy as np
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


SEVERITY_NAMES = ["Low", "Medium", "High"]


def compute_metrics(
    targets: List[int],
    predictions: List[int],
    beta: float = 2.0,
) -> Dict[str, float]:
    """
    Compute accuracy, weighted F1, and F-beta score.

    Args:
        targets     : Ground-truth class indices.
        predictions : Predicted class indices.
        beta        : Beta for F-beta score (default 2 = recall-heavy).

    Returns:
        Dict with keys: accuracy, f1_weighted, f_beta.
    """
    targets = np.array(targets)
    predictions = np.array(predictions)

    accuracy  = 100.0 * np.mean(targets == predictions)
    f1        = f1_score(targets, predictions, average="weighted", zero_division=0)
    f_beta    = fbeta_score(targets, predictions, beta=beta, average="weighted", zero_division=0)

    return {"accuracy": accuracy, "f1_weighted": f1, "f_beta": f_beta}


def print_classification_report(
    targets: List[int],
    predictions: List[int],
    label_names: List[str] = SEVERITY_NAMES,
) -> None:
    """Print a formatted sklearn classification report."""
    print(classification_report(targets, predictions, target_names=label_names, digits=3))


def get_confusion_matrix(
    targets: List[int],
    predictions: List[int],
) -> np.ndarray:
    """Return the confusion matrix as a numpy array."""
    return confusion_matrix(targets, predictions)


def compute_roc_data(
    targets: List[int],
    probabilities: np.ndarray,
    num_classes: int = 3,
) -> List[Dict]:
    """
    Compute one-vs-rest ROC curve data for each class.

    Returns:
        List of dicts: {class_name, fpr, tpr, auc}
    """
    targets = np.array(targets)
    roc_data = []
    for i, name in enumerate(SEVERITY_NAMES[:num_classes]):
        y_bin   = (targets == i).astype(int)
        y_score = probabilities[:, i]
        fpr, tpr, _ = roc_curve(y_bin, y_score)
        roc_auc = auc(fpr, tpr)
        roc_data.append({"class_name": name, "fpr": fpr, "tpr": tpr, "auc": roc_auc})
    return roc_data
