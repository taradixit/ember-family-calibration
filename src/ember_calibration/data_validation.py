"""Validation gates for metadata and detector predictions."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EXPECTED_PE_COUNTS = {"Win32": 360_000, "Win64": 120_000, "Dot_Net": 60_000}
EXPECTED_PE_RECORDS = sum(EXPECTED_PE_COUNTS.values())
REQUIRED_METADATA_COLUMNS = {"sha256", "label", "file_type", "family"}
REVIEWED_SELECTION_COLUMNS = REQUIRED_METADATA_COLUMNS | {"week_id"}
REVIEWED_UNIQUE_HASH_COUNT = 539_940
REVIEWED_HASH_MULTIPLICITIES = {1: 539_880, 2: 60}
REVIEWED_REPEATED_HASH_LIST_SHA256 = (
    "81c20f8d9397f4f27143652988dfdc036edc3a3c948a0efe75e7817e97283767"
)
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


def validate_reviewed_selection_metadata(
    metadata: pd.DataFrame,
    expected_records: int = EXPECTED_PE_RECORDS,
    expected_file_types: dict[str, int] = EXPECTED_PE_COUNTS,
    expected_unique_hashes: int = REVIEWED_UNIQUE_HASH_COUNT,
    expected_multiplicities: dict[int, int] = REVIEWED_HASH_MULTIPLICITIES,
    expected_repeated_hash_digest: str = REVIEWED_REPEATED_HASH_LIST_SHA256,
) -> ValidationReport:
    """Validate the narrow, reviewed 540,000-row official PE selection."""
    require_columns(metadata, REVIEWED_SELECTION_COLUMNS)
    report = build_report(metadata)
    errors: list[str] = []
    hashes = metadata["sha256"]
    if hashes.isna().any() or hashes.astype(str).str.strip().eq("").any():
        errors.append("missing or blank SHA-256 hashes")
    labels = metadata["label"]
    if labels.isna().any() or not labels.isin([0, 1]).all():
        errors.append("labels must contain only binary values 0 and 1")
    if report.row_count != expected_records:
        errors.append(f"expected {expected_records} selected PE records, found {report.row_count}")
    actual_types = {normalize_file_type(key): value for key, value in report.file_type_counts.items()}
    if actual_types != expected_file_types:
        errors.append(f"expected file-type counts {expected_file_types}, found {actual_types}")
    if report.unique_hash_count != expected_unique_hashes:
        errors.append(
            f"expected {expected_unique_hashes} unique selected hashes, "
            f"found {report.unique_hash_count}"
        )
    if report.duplicate_multiplicities != expected_multiplicities:
        errors.append(
            f"expected selected hash multiplicities {expected_multiplicities}, "
            f"found {report.duplicate_multiplicities}"
        )
    hash_counts = hashes.value_counts()
    repeated_hashes = sorted(str(value) for value in hash_counts[hash_counts > 1].index)
    repeated_digest = hashlib.sha256("\n".join(repeated_hashes).encode("utf-8")).hexdigest()
    if repeated_digest != expected_repeated_hash_digest:
        errors.append("selected repeated-hash list digest does not match the reviewed profile")
    repeated = metadata[metadata["sha256"].isin(repeated_hashes)]
    conflict_columns = ("label", "family", "file_type")
    for column in conflict_columns:
        conflicts = repeated.groupby("sha256", dropna=False)[column].nunique(dropna=False).gt(1).sum()
        if conflicts:
            errors.append(f"selected repeated hashes have {int(conflicts)} {column} conflicts")
    cross_week = repeated.groupby("sha256", dropna=False)["week_id"].nunique(dropna=False).gt(1).sum()
    if int(cross_week) != len(repeated_hashes):
        errors.append(
            "every reviewed repeated hash must cross week boundaries: "
            f"expected {len(repeated_hashes)}, found {int(cross_week)}"
        )
    if report.exact_duplicate_count:
        errors.append(f"found {report.exact_duplicate_count} exact selected metadata records")
    if errors:
        raise ValidationError("reviewed selection validation failed:\n- " + "\n- ".join(errors))
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
