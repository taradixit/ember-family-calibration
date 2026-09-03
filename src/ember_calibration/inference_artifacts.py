"""Validation for completed inference manifests and predictions."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import numpy as np

from .archive_manifest import sha256_file
from .inference import validate_prediction_array
from .preparation import EXPECTED_FEATURE_COUNT, EXPECTED_PREPARATION_ROWS
from .prepared_artifacts import load_prepared_artifacts
from .selection import repository_relative_path, resolve_repository_path
from .upstream import HISTORICAL_MODEL_SHA256, MODEL_REVISION

PREDICTION_MEANING = "released EMBER2024 PE-model malicious-class probability"


def _resolve_inference_path(manifest_path: Path, value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(value):
        raise ValueError(f"unsafe inference artifact path: {value}")
    resolved = manifest_path.parent.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError(f"inference artifact path escapes manifest directory: {value}")
    return resolved


def load_inference_artifacts(
    manifest_path: Path,
    repository_root: Path,
    expected_rows: int = EXPECTED_PREPARATION_ROWS,
    expected_features: int = EXPECTED_FEATURE_COUNT,
    expected_model_sha256: str = HISTORICAL_MODEL_SHA256,
    expected_model_revision: str = MODEL_REVISION,
    preparation_loader: Callable[..., dict[str, object]] = load_prepared_artifacts,
) -> dict[str, object]:
    repository_relative_path(manifest_path, repository_root)
    if manifest_path.name != "inference_manifest.json":
        raise ValueError("inference manifest has an unexpected filename")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("inference manifest has an unsupported schema version")
    if document.get("completion_status") != "complete":
        raise ValueError("inference manifest is incomplete")
    expected_path_bases = {
        "repository_root": "repository root",
        "inference_manifest_directory": "directory containing this manifest",
    }
    if document.get("path_bases") != expected_path_bases:
        raise ValueError("inference manifest has unsupported path bases")
    if document.get("rows") != expected_rows or document.get("feature_count") != expected_features:
        raise ValueError("inference dimensions do not match the reviewed preparation")
    if document.get("model_feature_count") != expected_features:
        raise ValueError("inference model feature count is incorrect")
    if document.get("prediction_meaning") != PREDICTION_MEANING:
        raise ValueError("inference prediction meaning is incorrect")
    if document.get("threshold_applied") is not False:
        raise ValueError("inference must not apply a classification threshold")
    batch_size = document.get("batch_size")
    batch_count = document.get("prediction_batch_count")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("inference manifest has an invalid batch size")
    expected_batches = (expected_rows + batch_size - 1) // batch_size
    if batch_count != expected_batches:
        raise ValueError("inference manifest has an invalid prediction batch count")
    if not isinstance(document.get("elapsed_seconds"), (int, float)) or document[
        "elapsed_seconds"
    ] < 0:
        raise ValueError("inference manifest has invalid elapsed time")
    if not isinstance(document.get("start_time_utc"), str) or not isinstance(
        document.get("finish_time_utc"), str
    ):
        raise ValueError("inference manifest has invalid UTC timestamps")
    if not re.fullmatch(r"[0-9a-f]{40}", str(document.get("code_commit"))):
        raise ValueError("inference manifest has an invalid code commit")
    if not isinstance(document.get("lightgbm_version"), str):
        raise ValueError("inference manifest has no LightGBM version")

    model_record = document.get("model")
    if not isinstance(model_record, dict) or model_record.get("path_base") != "repository_root":
        raise ValueError("inference manifest has an invalid model record")
    if (
        model_record.get("repository_revision") != expected_model_revision
        or model_record.get("sha256") != expected_model_sha256
    ):
        raise ValueError("inference manifest does not identify the pinned model")
    model_path = resolve_repository_path(model_record.get("path"), repository_root)
    if not model_path.is_file() or sha256_file(model_path) != expected_model_sha256:
        raise ValueError("model checksum mismatch")

    preparation_record = document.get("preparation_manifest")
    if (
        not isinstance(preparation_record, dict)
        or preparation_record.get("path_base") != "repository_root"
    ):
        raise ValueError("inference manifest has an invalid preparation-manifest record")
    preparation_path = resolve_repository_path(preparation_record.get("path"), repository_root)
    if not preparation_path.is_file() or sha256_file(preparation_path) != preparation_record.get(
        "sha256"
    ):
        raise ValueError("preparation-manifest checksum mismatch")
    prepared = preparation_loader(
        preparation_path,
        repository_root,
        expected_rows=expected_rows,
        expected_features=expected_features,
    )
    preparation_document = json.loads(preparation_path.read_text(encoding="utf-8"))
    inherited = document.get("preparation_artifact_sha256")
    expected_inherited = {
        name: preparation_document["artifacts"][name]["sha256"]
        for name in ("features", "labels", "metadata")
    }
    if inherited != expected_inherited:
        raise ValueError("inherited preparation artifact checksums do not match")

    command_arguments = document.get("command_arguments")
    if not isinstance(command_arguments, dict):
        raise ValueError("inference manifest has no normalized command arguments")
    for key in ("model", "preparation_manifest", "output_dir"):
        resolve_repository_path(command_arguments.get(key), repository_root)
    command_output_dir = resolve_repository_path(
        command_arguments.get("output_dir"), repository_root
    )
    if (
        command_arguments.get("model") != model_record.get("path")
        or command_arguments.get("preparation_manifest") != preparation_record.get("path")
        or command_output_dir != manifest_path.parent.resolve()
        or command_arguments.get("batch_size") != batch_size
        or command_arguments.get("execute") is not True
    ):
        raise ValueError("inference command arguments do not match its provenance")
    if not isinstance(command_arguments.get("overwrite"), bool):
        raise ValueError("inference overwrite argument is invalid")

    prediction_record = document.get("prediction")
    if (
        not isinstance(prediction_record, dict)
        or prediction_record.get("path_base") != "inference_manifest_directory"
    ):
        raise ValueError("inference manifest has an invalid prediction record")
    if prediction_record.get("path") != "predictions.npy":
        raise ValueError("inference manifest has an unexpected prediction path")
    prediction_path = _resolve_inference_path(manifest_path, prediction_record.get("path"))
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if prediction_path.stat().st_size != prediction_record.get("size_bytes"):
        raise ValueError("prediction file size does not match inference manifest")
    if sha256_file(prediction_path) != prediction_record.get("sha256"):
        raise ValueError("prediction checksum mismatch")
    if (
        prediction_record.get("filename") != "predictions.npy"
        or prediction_record.get("dtype") != "float64"
        or prediction_record.get("shape") != [expected_rows]
        or prediction_record.get("finite_values") is not True
        or prediction_record.get("range_valid") is not True
        or prediction_record.get("exact_row_count") is not True
    ):
        raise ValueError("prediction validation record is incomplete")
    actual_names = {path.name for path in manifest_path.parent.iterdir()}
    if actual_names != {"predictions.npy", "inference_manifest.json"}:
        raise ValueError("completed inference directory has unexpected contents")
    predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    try:
        validated = validate_prediction_array(predictions, expected_rows, require_float64=True)
        if (
            float(validated.min()) != prediction_record.get("minimum")
            or float(validated.max()) != prediction_record.get("maximum")
        ):
            raise ValueError("prediction range does not match inference manifest")
    finally:
        del predictions
    return {
        "rows": expected_rows,
        "feature_count": expected_features,
        "predictions": prediction_path,
        "metadata": prepared["metadata"],
        "preparation_manifest": preparation_path,
        "model": model_path,
    }
