"""Bounded prediction validation and transactional publication helpers."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_BATCH_SIZE = 10_000
PREDICTION_DTYPE = np.dtype(np.float64)


def positive_batch_size(value: str | int) -> int:
    if isinstance(value, bool):
        raise ValueError("batch size must be a positive integer")
    if isinstance(value, int):
        batch_size = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        batch_size = int(value)
    else:
        raise ValueError("batch size must be a positive integer")
    if batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    return batch_size


def validate_prediction_array(
    predictions: object,
    expected_rows: int,
    require_float64: bool = False,
) -> np.ndarray:
    values = np.asarray(predictions)
    if values.ndim != 1:
        raise ValueError("predictions must be one-dimensional")
    if values.size != expected_rows:
        raise ValueError(
            f"prediction count mismatch: expected {expected_rows}, found {values.size}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("predictions must have a numeric dtype")
    if require_float64 and values.dtype != PREDICTION_DTYPE:
        raise ValueError("prediction dtype must be float64")
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("predictions must be in [0, 1]")
    return values


def write_batched_predictions(
    feature_path: Path,
    prediction_path: Path,
    model: object,
    rows: int,
    feature_count: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    positive_batch_size(batch_size)
    if rows < 1 or feature_count < 1:
        raise ValueError("prediction dimensions must be positive")
    features = np.memmap(
        feature_path,
        dtype=np.float32,
        mode="r",
        shape=(rows, feature_count),
    )
    predictions = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=PREDICTION_DTYPE,
        shape=(rows,),
    )
    predictions[:] = np.nan
    first_row_written = False
    last_row_written = False
    batches = 0
    try:
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            batch_predictions = validate_prediction_array(
                model.predict(features[start:stop]),
                stop - start,
            )
            predictions[start:stop] = batch_predictions
            first_row_written = first_row_written or start == 0
            last_row_written = last_row_written or stop == rows
            batches += 1
        if not first_row_written or not last_row_written:
            raise ValueError("first or last prediction row was not written")
        predictions.flush()
    finally:
        del predictions
        del features

    reloaded = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    try:
        validated = validate_prediction_array(reloaded, rows, require_float64=True)
        minimum = float(validated.min())
        maximum = float(validated.max())
        shape = list(validated.shape)
    finally:
        del reloaded
    return {
        "batch_count": batches,
        "dtype": "float64",
        "shape": shape,
        "size_bytes": prediction_path.stat().st_size,
        "minimum": minimum,
        "maximum": maximum,
        "finite_values": True,
        "range_valid": True,
        "exact_row_count": True,
        "first_row_written": True,
        "last_row_written": True,
    }


def _operation_prefix(output_dir: Path, operation: str) -> str:
    return f".{output_dir.name}.{operation}-"


def create_inference_staging_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=_operation_prefix(output_dir, "staging"),
            dir=output_dir.parent,
        )
    )


def reserve_inference_backup_path(output_dir: Path) -> Path:
    backup_path = Path(
        tempfile.mkdtemp(
            prefix=_operation_prefix(output_dir, "backup"),
            dir=output_dir.parent,
        )
    )
    backup_path.rmdir()
    return backup_path


def remove_inference_operation_directory(
    path: Path,
    output_dir: Path,
    operation: str,
) -> None:
    if not path.exists():
        return
    if path.parent.resolve() != output_dir.parent.resolve() or not path.name.startswith(
        _operation_prefix(output_dir, operation)
    ):
        raise ValueError(f"refusing to remove unexpected inference {operation} directory")
    shutil.rmtree(path)


def rename_inference_directory(source: Path, destination: Path) -> None:
    source.replace(destination)


def publish_inference_directory(staging_dir: Path, output_dir: Path, overwrite: bool) -> None:
    backup_path: Path | None = None
    previous_moved = False
    if output_dir.exists():
        if not overwrite:
            raise ValueError("completed inference already exists; pass --overwrite to replace it")
        backup_path = reserve_inference_backup_path(output_dir)
        rename_inference_directory(output_dir, backup_path)
        previous_moved = True
    try:
        rename_inference_directory(staging_dir, output_dir)
    except Exception:
        if output_dir.exists() and not staging_dir.exists():
            rename_inference_directory(output_dir, staging_dir)
        if previous_moved and backup_path is not None and not output_dir.exists():
            rename_inference_directory(backup_path, output_dir)
        raise
    if backup_path is not None:
        remove_inference_operation_directory(backup_path, output_dir, "backup")
