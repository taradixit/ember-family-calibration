"""Small, dependency-light metric implementations used by the analysis."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score

CALIBRATION_THRESHOLD = 0.5
MINIMUM_PREDICTED_LABEL_CONFIDENCE = 0.5
MAXIMUM_PREDICTED_LABEL_CONFIDENCE = 1.0


def _as_labels(values: Iterable[int]) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must contain only 0 and 1")
    return labels.astype(int)


def _as_predictions(values: Iterable[float]) -> np.ndarray:
    predictions = np.asarray(values, dtype=float)
    if predictions.ndim != 1 or predictions.size == 0:
        raise ValueError("predictions must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("predictions must all be finite")
    if np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("predictions must be between 0 and 1 inclusive")
    return predictions


def _checked_arrays(labels: Iterable[int], predictions: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    label_array = _as_labels(labels)
    prediction_array = _as_predictions(predictions)
    if label_array.size != prediction_array.size:
        raise ValueError("labels and predictions must have the same length")
    return label_array, prediction_array


def _check_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")


def predicted_labels(predictions: Iterable[float], threshold: float = 0.5) -> np.ndarray:
    _check_threshold(threshold)
    return (_as_predictions(predictions) >= threshold).astype(int)


def accuracy(labels: Iterable[int], predictions: Iterable[float], threshold: float = 0.5) -> float:
    label_array, prediction_array = _checked_arrays(labels, predictions)
    _check_threshold(threshold)
    return float(np.mean((prediction_array >= threshold) == label_array))


def roc_auc(labels: Iterable[int], predictions: Iterable[float]) -> float:
    """Return binary ROC AUC using scikit-learn's efficient implementation."""
    label_array, prediction_array = _checked_arrays(labels, predictions)
    if np.unique(label_array).size != 2:
        raise ValueError("ROC AUC requires at least one positive and one negative label")
    return float(roc_auc_score(label_array, prediction_array))


def brier_score(labels: Iterable[int], predictions: Iterable[float]) -> float:
    label_array, prediction_array = _checked_arrays(labels, predictions)
    return float(np.mean((prediction_array - label_array) ** 2))


def predicted_label_confidence(predictions: Iterable[float]) -> np.ndarray:
    """Return confidence for the class predicted at the standard 0.5 threshold."""
    prediction_array = _as_predictions(predictions)
    return np.maximum(prediction_array, 1.0 - prediction_array)


def reliability_bins(
    labels: Iterable[int],
    predictions: Iterable[float],
    n_bins: int = 15,
) -> list[dict[str, float | int | None]]:
    """Summarize equal-width bins over [0.5, 1.0]; the final bin includes 1.0."""
    label_array, prediction_array = _checked_arrays(labels, predictions)
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    confidence = predicted_label_confidence(prediction_array)
    correct = ((prediction_array >= CALIBRATION_THRESHOLD) == label_array).astype(float)
    edges = np.linspace(MINIMUM_PREDICTED_LABEL_CONFIDENCE, MAXIMUM_PREDICTED_LABEL_CONFIDENCE, n_bins + 1)
    indexes = np.searchsorted(edges, confidence, side="right") - 1
    indexes = np.minimum(indexes, n_bins - 1)
    summaries: list[dict[str, float | int | None]] = []
    for index in range(n_bins):
        selected = indexes == index
        count = int(np.sum(selected))
        mean_confidence = float(np.mean(confidence[selected])) if count else None
        bin_accuracy = float(np.mean(correct[selected])) if count else None
        gap = abs(bin_accuracy - mean_confidence) if count else None
        summaries.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": bin_accuracy,
                "gap": gap,
            }
        )
    return summaries


def expected_calibration_error(
    labels: Iterable[int], predictions: Iterable[float], n_bins: int = 15
) -> float:
    bins = reliability_bins(labels, predictions, n_bins)
    total = sum(int(item["count"]) for item in bins)
    return float(sum(int(item["count"]) * float(item["gap"] or 0.0) for item in bins) / total)


def maximum_calibration_error(
    labels: Iterable[int], predictions: Iterable[float], n_bins: int = 15
) -> float:
    gaps = [float(item["gap"]) for item in reliability_bins(labels, predictions, n_bins) if item["gap"] is not None]
    return max(gaps)
