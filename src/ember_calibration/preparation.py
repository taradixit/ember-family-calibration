"""Testable checks and publication helpers for prepared artifacts."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_PREPARATION_ROWS = 540_000
EXPECTED_FEATURE_COUNT = 2_568
FEATURE_DTYPE = np.dtype(np.float32)
LABEL_DTYPE = np.dtype(np.int32)
EXPECTED_FEATURE_BYTES = EXPECTED_PREPARATION_ROWS * EXPECTED_FEATURE_COUNT * FEATURE_DTYPE.itemsize
EXPECTED_LABEL_BYTES = EXPECTED_PREPARATION_ROWS * LABEL_DTYPE.itemsize
METADATA_COLUMNS = ["sha256", "label", "file_type", "family", "week_id"]


def validate_feature_file(
    path: Path,
    expected_rows: int = EXPECTED_PREPARATION_ROWS,
    expected_features: int = EXPECTED_FEATURE_COUNT,
    chunk_rows: int = 4_096,
) -> dict[str, object]:
    if expected_rows < 1 or expected_features < 1 or chunk_rows < 1:
        raise ValueError("feature validation dimensions must be positive")
    expected_size = expected_rows * expected_features * FEATURE_DTYPE.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"feature file size mismatch: expected {expected_size} bytes, found {actual_size}"
        )
    features = np.memmap(
        path,
        dtype=FEATURE_DTYPE,
        mode="r",
        shape=(expected_rows, expected_features),
    )
    try:
        for start in range(0, expected_rows, chunk_rows):
            stop = min(start + chunk_rows, expected_rows)
            if not np.isfinite(features[start:stop]).all():
                raise ValueError("feature file contains NaN or infinity")
    finally:
        del features
    return {
        "rows": expected_rows,
        "feature_count": expected_features,
        "dtype": "float32",
        "size_bytes": actual_size,
        "finite_values": True,
    }


def validate_vectorized_label_file(
    path: Path,
    expected_labels: Iterable[int],
    expected_rows: int | None = None,
) -> np.ndarray:
    labels = np.asarray(expected_labels, dtype=LABEL_DTYPE)
    required_rows = labels.size if expected_rows is None else expected_rows
    if labels.size != required_rows:
        raise ValueError(
            f"metadata label count mismatch: expected {required_rows}, found {labels.size}"
        )
    expected_size = required_rows * LABEL_DTYPE.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"label file size mismatch: expected {expected_size} bytes, found {actual_size}")
    vectorized = np.memmap(path, dtype=LABEL_DTYPE, mode="r", shape=(required_rows,))
    try:
        if not np.all(np.isin(vectorized, (0, 1))):
            raise ValueError(
                "vectorized labels contain values other than 0 and 1; expected int32 encoding"
            )
        if not np.array_equal(vectorized, labels):
            raise ValueError("vectorized labels are not aligned with metadata rows")
    finally:
        del vectorized
    return labels


def validate_metadata_file(
    path: Path,
    expected_metadata: pd.DataFrame,
    metadata_validator: Callable[[pd.DataFrame], object],
    expected_rows: int = EXPECTED_PREPARATION_ROWS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if list(expected_metadata.columns) != METADATA_COLUMNS:
        raise ValueError(f"metadata columns must be exactly {METADATA_COLUMNS}")
    if len(expected_metadata) != expected_rows:
        raise ValueError(
            f"metadata row count mismatch: expected {expected_rows}, found {len(expected_metadata)}"
        )
    observed = pd.read_parquet(path)
    if list(observed.columns) != METADATA_COLUMNS:
        raise ValueError(f"metadata parquet columns must be exactly {METADATA_COLUMNS}")
    if len(observed) != expected_rows:
        raise ValueError(
            f"metadata parquet row count mismatch: expected {expected_rows}, found {len(observed)}"
        )
    if not observed.equals(expected_metadata.reset_index(drop=True)):
        raise ValueError("metadata parquet does not preserve selected row order and values")
    metadata_validator(observed)
    return observed, {
        "rows": expected_rows,
        "columns": METADATA_COLUMNS,
        "selected_order_preserved": True,
        "reviewed_repeat_profile": True,
    }


def _operation_prefix(output_dir: Path, operation: str) -> str:
    return f".{output_dir.name}.{operation}-"


def create_preparation_staging_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=_operation_prefix(output_dir, "staging"),
            dir=output_dir.parent,
        )
    )


def reserve_preparation_backup_path(output_dir: Path) -> Path:
    backup_path = Path(
        tempfile.mkdtemp(
            prefix=_operation_prefix(output_dir, "backup"),
            dir=output_dir.parent,
        )
    )
    backup_path.rmdir()
    return backup_path


def remove_preparation_operation_directory(
    path: Path,
    output_dir: Path,
    operation: str,
) -> None:
    if not path.exists():
        return
    if path.parent.resolve() != output_dir.parent.resolve() or not path.name.startswith(
        _operation_prefix(output_dir, operation)
    ):
        raise ValueError(f"refusing to remove unexpected preparation {operation} directory")
    shutil.rmtree(path)


def rename_preparation_directory(source: Path, destination: Path) -> None:
    source.replace(destination)


def publish_prepared_directory(staging_dir: Path, output_dir: Path, overwrite: bool) -> None:
    backup_path: Path | None = None
    previous_moved = False
    if output_dir.exists():
        if not overwrite:
            raise ValueError("completed preparation already exists; pass --overwrite to replace it")
        backup_path = reserve_preparation_backup_path(output_dir)
        rename_preparation_directory(output_dir, backup_path)
        previous_moved = True
    try:
        rename_preparation_directory(staging_dir, output_dir)
    except Exception:
        if output_dir.exists() and not staging_dir.exists():
            rename_preparation_directory(output_dir, staging_dir)
        if previous_moved and backup_path is not None and not output_dir.exists():
            rename_preparation_directory(backup_path, output_dir)
        raise
    if backup_path is not None:
        remove_preparation_operation_directory(backup_path, output_dir, "backup")
