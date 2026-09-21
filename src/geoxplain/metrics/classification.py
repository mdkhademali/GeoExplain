"""Classification metrics (binary and multi-class)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: np.ndarray | None = None,
    classes: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, ROC-AUC, PR-AUC (+ balanced accuracy).

    Binary problems use the second class (sorted order) as the positive class.
    Multi-class problems use *macro* averaging for precision/recall/F1 and one-vs-rest
    macro averaging for ROC-AUC and PR-AUC. AUC values are ``NaN`` when ``y_proba`` is not
    provided or when ``y_true`` contains a single class (AUC undefined).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.asarray(classes if classes is not None else np.unique(np.concatenate([y_true, y_pred])))
    binary = len(labels) == 2
    average = "binary" if binary else "macro"
    kw: dict[str, Any] = {"average": average, "zero_division": 0}
    if binary:
        kw["pos_label"] = labels[1]
    else:
        kw["labels"] = labels
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **kw)),
        "recall": float(recall_score(y_true, y_pred, **kw)),
        "f1": float(f1_score(y_true, y_pred, **kw)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        proba = np.asarray(y_proba, dtype=float)
        try:
            if binary:
                pos = (y_true == labels[1]).astype(int)
                out["roc_auc"] = float(roc_auc_score(pos, proba[:, 1]))
                out["pr_auc"] = float(average_precision_score(pos, proba[:, 1]))
            else:
                present = [i for i, c in enumerate(labels) if np.any(y_true == c)]
                onehot = label_binarize(y_true, classes=labels)
                out["roc_auc"] = float(
                    roc_auc_score(onehot[:, present], proba[:, present], average="macro")
                )
                out["pr_auc"] = float(
                    average_precision_score(onehot[:, present], proba[:, present], average="macro")
                )
        except ValueError:
            pass
    return out


def confusion(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> np.ndarray:
    """Confusion matrix with rows = true class and columns = predicted class."""
    return confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=list(labels))


def roc_curve_data(y_true: Sequence[Any], y_score: np.ndarray, positive_label: Any) -> tuple[np.ndarray, np.ndarray]:
    """False/true positive rates for a one-vs-rest ROC curve."""
    pos = (np.asarray(y_true) == positive_label).astype(int)
    fpr, tpr, _ = roc_curve(pos, y_score)
    return fpr, tpr


def pr_curve_data(y_true: Sequence[Any], y_score: np.ndarray, positive_label: Any) -> tuple[np.ndarray, np.ndarray]:
    """Recall/precision arrays for a one-vs-rest precision-recall curve."""
    pos = (np.asarray(y_true) == positive_label).astype(int)
    precision, recall, _ = precision_recall_curve(pos, y_score)
    return recall, precision
