"""Family-level summaries for malicious examples."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .data_validation import ValidationError, filter_by_minimum_count
from .metrics import accuracy, expected_calibration_error


def family_metrics(
    metadata: pd.DataFrame,
    predictions: Iterable[float],
    minimum_count: int = 100,
    threshold: float = 0.5,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Return metrics for qualifying malicious-only family groups.

    Because every included label is 1, false-negative rate equals 1 - accuracy.
    """
    values = np.asarray(predictions, dtype=float)
    if len(metadata) != values.size:
        raise ValidationError("prediction/metadata length mismatch")
    required = {"family", "label"}
    if not required.issubset(metadata.columns):
        raise ValidationError(f"missing family-analysis columns: {sorted(required - set(metadata.columns))}")
    working = metadata.copy()
    working["prediction"] = values
    malicious = working[working["label"] == 1].copy()
    normalized_families = malicious["family"].astype("string").str.strip().str.casefold()
    unusable = normalized_families.isna() | normalized_families.isin({"", "unknown", "none", "nan"})
    malicious = malicious[~unusable].copy()
    eligible = filter_by_minimum_count(malicious, "family", minimum_count)
    rows = []
    for family, group in eligible.groupby("family", sort=True):
        labels = group["label"].to_numpy()
        scores = group["prediction"].to_numpy()
        family_accuracy = accuracy(labels, scores, threshold)
        rows.append(
            {
                "family": family,
                "count": len(group),
                "accuracy": family_accuracy,
                "false_negative_rate": 1.0 - family_accuracy,
                "ece": expected_calibration_error(labels, scores, n_bins),
            }
        )
    result = pd.DataFrame(rows, columns=["family", "count", "accuracy", "false_negative_rate", "ece"])
    return result.sort_values(
        ["false_negative_rate", "ece", "family"], ascending=[False, False, True], ignore_index=True
    )
