"""Testable checks for vectorized preparation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def validate_vectorized_label_file(path: Path, expected_labels: Iterable[int]) -> np.ndarray:
    labels = np.asarray(expected_labels, dtype=np.int32)
    expected_size = labels.size * np.dtype(np.int32).itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"label file size mismatch: expected {expected_size} bytes, found {actual_size}")
    vectorized = np.memmap(path, dtype=np.int32, mode="r", shape=(labels.size,))
    if not np.all(np.isin(vectorized, (0, 1))):
        raise ValueError("vectorized labels contain values other than 0 and 1; expected int32 encoding")
    if not np.array_equal(vectorized, labels):
        raise ValueError("vectorized labels are not aligned with metadata rows")
    return vectorized

