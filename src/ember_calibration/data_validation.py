"""Validation gates for metadata and detector predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EXPECTED_PE_COUNTS = {"Win32": 360_000, "Win64": 120_000, "Dot_Net": 60_000}
EXPECTED_PE_RECORDS = sum(EXPECTED_PE_COUNTS.values())
REQUIRED_METADATA_COLUMNS = {"sha256", "label", "file_type", "family"}
FILE_TYPE_ALIASES = {".NET": "Dot_Net", "DotNet": "Dot_Net", "dotnet": "Dot_Net"}


class ValidationError(ValueError):
    """Raised when an input fails a reproducibility validation gate."""


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    unique_hash_count: int
    duplicate_multiplicities: dict[int, int]
    exact_duplicate_count: int
    label_counts: dict[object, int]
    file_type_counts: dict[str, int]

    def as_text(self) -> str:
        return "\n".join(
            [
                f"row count: {self.row_count}",
                f"unique hash count: {self.unique_hash_count}",
                f"duplicate multiplicities (copies -> hashes): {self.duplicate_multiplicities}",
                f"exact duplicate records: {self.exact_duplicate_count}",
                f"label counts: {self.label_counts}",
                f"file-type counts: {self.file_type_counts}",
            ]
        )


def require_columns(metadata: pd.DataFrame, required: set[str] = REQUIRED_METADATA_COLUMNS) -> None:
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValidationError(f"missing required metadata columns: {missing}")


def normalize_file_type(value: object) -> str:
    text = str(value)
    return FILE_TYPE_ALIASES.get(text, text)


def build_report(metadata: pd.DataFrame) -> ValidationReport:
    require_columns(metadata)
    hashes = metadata["sha256"]
    hash_counts = hashes.dropna().value_counts()
    multiplicities = Counter(int(value) for value in hash_counts.values)
    normalized_types = metadata["file_type"].map(normalize_file_type)
    return ValidationReport(
        row_count=len(metadata),
        unique_hash_count=int(hashes.nunique(dropna=True)),
        duplicate_multiplicities=dict(sorted(multiplicities.items())),
        exact_duplicate_count=int(metadata.duplicated().sum()),
        label_counts=metadata["label"].value_counts(dropna=False).to_dict(),
        file_type_counts=normalized_types.value_counts(dropna=False).to_dict(),
    )


def validate_predictions(predictions: Iterable[float], expected_length: int | None = None) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1:
        raise ValidationError("predictions must be one-dimensional")
    if expected_length is not None and values.size != expected_length:
        raise ValidationError(f"prediction/metadata length mismatch: {values.size} != {expected_length}")
    if not np.all(np.isfinite(values)):
        raise ValidationError("predictions contain missing or non-finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValidationError("predictions must be between 0 and 1 inclusive")
    return values


def validate_metadata(
    metadata: pd.DataFrame,
    expected_records: int = EXPECTED_PE_RECORDS,
    expected_file_types: dict[str, int] = EXPECTED_PE_COUNTS,
) -> ValidationReport:
    report = build_report(metadata)
    errors: list[str] = []
    if metadata["sha256"].isna().any() or metadata["sha256"].astype(str).str.strip().eq("").any():
        errors.append("missing or blank SHA-256 hashes")
    labels = metadata["label"]
    if labels.isna().any() or not labels.isin([0, 1]).all():
        errors.append("labels must contain only binary values 0 and 1")
    if report.row_count != expected_records:
        errors.append(f"expected {expected_records} PE records, found {report.row_count}")
    actual_types = {normalize_file_type(key): value for key, value in report.file_type_counts.items()}
    if actual_types != expected_file_types:
        errors.append(f"expected file-type counts {expected_file_types}, found {actual_types}")
    duplicated = metadata["sha256"].duplicated(keep=False)
    if duplicated.any():
        errors.append(f"duplicate SHA-256 hashes detected in {int(duplicated.sum())} rows; no deduplication was performed")
    if report.exact_duplicate_count:
        errors.append(f"found {report.exact_duplicate_count} exact duplicate records")
    if report.row_count == 2 * expected_records and report.unique_hash_count <= expected_records:
        errors.append("dataset appears systematically doubled")
    if errors:
        raise ValidationError("validation failed:\n- " + "\n- ".join(errors) + "\n\n" + report.as_text())
    return report


def validate_inputs(metadata: pd.DataFrame, predictions: Iterable[float]) -> ValidationReport:
    report = validate_metadata(metadata)
    validate_predictions(predictions, expected_length=len(metadata))
    return report


def filter_by_minimum_count(metadata: pd.DataFrame, column: str, minimum_count: int) -> pd.DataFrame:
    if column not in metadata.columns:
        raise ValidationError(f"missing grouping column: {column}")
    if not isinstance(minimum_count, int) or isinstance(minimum_count, bool) or minimum_count < 1:
        raise ValidationError("minimum_count must be a positive integer")
    counts = metadata[column].value_counts(dropna=False)
    keep = set(counts[counts >= minimum_count].index)
    return metadata[metadata[column].isin(keep)].copy()

