"""Validation for artifacts recorded by the preparation stage."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .archive_manifest import sha256_file
from .data_validation import validate_reviewed_selection_metadata
from .preparation import (
    EXPECTED_FEATURE_COUNT,
    EXPECTED_PREPARATION_ROWS,
    METADATA_COLUMNS,
    validate_feature_file,
    validate_vectorized_label_file,
)
from .selection import (
    EXPECTED_THREMBER_VERSION,
    REPEATED_HASH_LIST_SHA256,
    SELECTION_RULE,
    resolve_repository_path,
)


def _resolve_recorded_path(manifest_path: Path, value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(value):
        raise ValueError(f"unsafe artifact path in preparation manifest: {value}")
    resolved = manifest_path.parent.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError(f"artifact path escapes preparation directory: {value}")
    return resolved


def load_prepared_artifacts(
    manifest_path: Path,
    repository_root: Path,
    expected_rows: int = EXPECTED_PREPARATION_ROWS,
    expected_features: int = EXPECTED_FEATURE_COUNT,
    metadata_validator: Callable[[pd.DataFrame], object] = validate_reviewed_selection_metadata,
) -> dict[str, object]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 2:
        raise ValueError("preparation manifest has an unsupported schema version")
    if document.get("completion_status") != "complete":
        raise ValueError("preparation manifest is incomplete")
    expected_path_bases = {
        "repository_root": "repository root",
        "preparation_manifest_directory": "directory containing this manifest",
    }
    if document.get("path_bases") != expected_path_bases:
        raise ValueError("preparation manifest has unsupported path bases")
    rows = document.get("rows")
    feature_count = document.get("feature_count")
    if rows != expected_rows:
        raise ValueError(f"preparation manifest row count must equal {expected_rows}")
    if feature_count != expected_features:
        raise ValueError(f"preparation manifest feature count must equal {expected_features}")
    if document.get("feature_dtype") != "float32" or document.get("label_dtype") != "int32":
        raise ValueError("preparation manifest has incorrect artifact dtypes")
    if not isinstance(document.get("elapsed_seconds"), (int, float)) or document["elapsed_seconds"] < 0:
        raise ValueError("preparation manifest has invalid elapsed time")
    if not isinstance(document.get("start_time_utc"), str) or not isinstance(
        document.get("finish_time_utc"), str
    ):
        raise ValueError("preparation manifest has invalid UTC timestamps")
    if not re.fullmatch(r"[0-9a-f]{40}", str(document.get("code_commit"))):
        raise ValueError("preparation manifest has an invalid code commit")
    command_arguments = document.get("command_arguments")
    if not isinstance(command_arguments, dict):
        raise ValueError("preparation manifest has no normalized command arguments")
    for key in ("selection_manifest", "output_dir"):
        resolve_repository_path(command_arguments.get(key), repository_root)
    if command_arguments.get("execute_vectorization") is not True:
        raise ValueError("preparation manifest does not record vectorization execution")
    dependencies = document.get("dependency_versions")
    if not isinstance(dependencies, dict) or any(
        not isinstance(dependencies.get(name), str)
        for name in ("numpy", "pandas", "pyarrow", "thrember")
    ):
        raise ValueError("preparation manifest has incomplete dependency versions")
    extractor = document.get("extractor")
    if not isinstance(extractor, dict):
        raise ValueError("preparation manifest has no extractor record")
    if extractor.get("version") != EXPECTED_THREMBER_VERSION:
        raise ValueError("preparation manifest has the wrong extractor version")
    if extractor.get("dimension") != expected_features:
        raise ValueError("preparation manifest has the wrong extractor dimension")

    selection_record = document.get("selection_manifest")
    if not isinstance(selection_record, dict):
        raise ValueError("preparation manifest has no selection-manifest record")
    if selection_record.get("path_base") != "repository_root":
        raise ValueError("selection manifest has an unsupported path base")
    selection_path = resolve_repository_path(selection_record.get("path"), repository_root)
    if not selection_path.is_file() or sha256_file(selection_path) != selection_record.get("sha256"):
        raise ValueError("selection-manifest checksum mismatch")
    selection_document = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection_record.get("selection_rule_name") != SELECTION_RULE
        or selection_document.get("selection_rule_name") != SELECTION_RULE
    ):
        raise ValueError("preparation manifest has the wrong selection rule")
    if (
        selection_record.get("repeated_hash_list_sha256") != REPEATED_HASH_LIST_SHA256
        or selection_document.get("repeated_hash_list_sha256") != REPEATED_HASH_LIST_SHA256
    ):
        raise ValueError("preparation manifest has the wrong repeated-hash-list digest")

    inputs = document.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("preparation manifest has no selected input records")
    selected_members = selection_document.get("selected_member_order")
    if not isinstance(selected_members, list) or len(inputs) != len(selected_members):
        raise ValueError("prepared inputs do not match the reviewed selection manifest")
    for record, selected_member in zip(inputs, selected_members, strict=True):
        if not isinstance(record, dict) or record.get("path_base") != "repository_root":
            raise ValueError("prepared input has an unsupported path base")
        if not isinstance(selected_member, dict):
            raise ValueError("selection manifest has an invalid selected-member record")
        expected_input = {
            "file_type": selected_member.get("file_type"),
            "path": selected_member.get("selected_output_path"),
            "path_base": "repository_root",
            "rows": selected_member.get("selected_row_count"),
            "sha256": selected_member.get("selected_output_sha256"),
        }
        if record != expected_input:
            raise ValueError("prepared input order or metadata differs from reviewed selection")
        input_path = resolve_repository_path(record.get("path"), repository_root)
        if not input_path.is_file() or sha256_file(input_path) != record.get("sha256"):
            raise ValueError("prepared input checksum mismatch")

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preparation manifest has no artifact records")
    resolved: dict[str, Path] = {}
    for name in ("features", "labels", "metadata"):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"missing {name} artifact record")
        if record.get("path_base") != "preparation_manifest_directory":
            raise ValueError(f"{name} artifact has an unsupported path base")
        path = _resolve_recorded_path(manifest_path, record.get("path"))
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"{name} file size does not match preparation manifest")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{name} checksum does not match preparation manifest")
        resolved[name] = path
    expected_feature_bytes = expected_rows * expected_features * np.dtype(np.float32).itemsize
    if resolved["features"].stat().st_size != expected_feature_bytes:
        raise ValueError(
            f"feature file byte-size mismatch: expected {expected_feature_bytes}, "
            f"found {resolved['features'].stat().st_size}"
        )
    expected_label_bytes = expected_rows * np.dtype(np.int32).itemsize
    if resolved["labels"].stat().st_size != expected_label_bytes:
        raise ValueError(
            f"label file byte-size mismatch: expected {expected_label_bytes}, "
            f"found {resolved['labels'].stat().st_size}"
        )
    feature_record = artifacts["features"]
    if (
        feature_record.get("rows") != expected_rows
        or feature_record.get("feature_count") != expected_features
        or feature_record.get("dtype") != "float32"
        or feature_record.get("finite_values") is not True
    ):
        raise ValueError("feature validation record is incomplete")
    label_record = artifacts["labels"]
    if (
        label_record.get("count") != expected_rows
        or label_record.get("dtype") != "int32"
        or label_record.get("allowed_values") is not True
        or label_record.get("aligned_with_metadata") is not True
    ):
        raise ValueError("label validation record is incomplete")
    metadata_record = artifacts["metadata"]
    if (
        metadata_record.get("rows") != expected_rows
        or metadata_record.get("columns") != METADATA_COLUMNS
        or metadata_record.get("labels_aligned") is not True
        or metadata_record.get("selected_order_preserved") is not True
        or metadata_record.get("reviewed_repeat_profile") is not True
    ):
        raise ValueError("metadata validation record is incomplete")
    validate_feature_file(
        resolved["features"],
        expected_rows=expected_rows,
        expected_features=expected_features,
    )
    metadata = pd.read_parquet(resolved["metadata"])
    if list(metadata.columns) != METADATA_COLUMNS or len(metadata) != expected_rows:
        raise ValueError("prepared metadata shape does not match the manifest")
    metadata_validator(metadata)
    validate_vectorized_label_file(
        resolved["labels"],
        metadata["label"].to_numpy(dtype=np.int32),
        expected_rows=expected_rows,
    )
    return {
        "rows": rows,
        "feature_count": feature_count,
        "metadata": resolved["metadata"],
        "features": resolved["features"],
        "labels": resolved["labels"],
        "selection_manifest": selection_path,
    }
