#!/usr/bin/env python3
"""Validate a completed reviewed selection before vectorization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.archive_manifest import sha256_file  # noqa: E402
from ember_calibration.data_validation import validate_reviewed_selection_metadata  # noqa: E402
from ember_calibration.preparation import (  # noqa: E402
    EXPECTED_FEATURE_COUNT,
    EXPECTED_PREPARATION_ROWS,
    METADATA_COLUMNS,
    create_preparation_staging_directory,
    publish_prepared_directory,
    remove_preparation_operation_directory,
    validate_feature_file,
    validate_metadata_file,
    validate_vectorized_label_file,
)
from ember_calibration.prepared_artifacts import load_prepared_artifacts  # noqa: E402
from ember_calibration.selection import (  # noqa: E402
    load_completed_selection_manifest,
    repository_relative_path,
)

EXPECTED_INPUT_COUNTS = {"Win32": 360_000, "Win64": 120_000, "Dot_Net": 60_000}


def validate_aggregate_record_counts(
    observed: Counter[str], expected: dict[str, int] = EXPECTED_INPUT_COUNTS
) -> None:
    if dict(observed) != expected:
        raise ValueError(
            f"aggregate file-type record counts mismatch: expected {expected}, found {dict(observed)}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-vectorization", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.overwrite and not args.execute_vectorization:
        parser.error("--overwrite requires --execute-vectorization")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_commit(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def dependency_versions() -> dict[str, str]:
    return {
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "pyarrow": version("pyarrow"),
        "thrember": version("thrember"),
    }


def inspect_jsonl(path: Path) -> tuple[int, list[dict[str, object]]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            rows.append(
                {
                    "sha256": record.get("sha256"),
                    "label": record.get("label"),
                    "file_type": record.get("file_type"),
                    "family": record.get("family"),
                    "week_id": record.get("week_id"),
                }
            )
    return len(rows), rows


def build_preparation_provenance(
    repository_root: Path,
    selection_manifest: Path,
    selection_document: dict[str, object],
    input_observations: list[dict[str, object]],
    row_count: int,
    feature_count: int,
    feature_path: Path,
    label_path: Path,
    metadata_path: Path,
    feature_validation: dict[str, object],
    label_validation: dict[str, object],
    metadata_validation: dict[str, object],
    start_time: str,
    finish_time: str,
    elapsed_seconds: float,
    code_commit: str,
    command_arguments: dict[str, object],
    versions: dict[str, str],
) -> dict[str, object]:
    output_dir = feature_path.parent.resolve()
    repository_relative_path(output_dir, repository_root)
    for artifact_path in (label_path, metadata_path):
        if artifact_path.parent.resolve() != output_dir:
            raise ValueError("preparation artifacts must share one output directory")
    input_records = []
    for observation in input_observations:
        input_path = Path(observation["path"])
        input_records.append(
            {
                "file_type": observation["file_type"],
                "path": repository_relative_path(input_path, repository_root),
                "path_base": "repository_root",
                "rows": observation["rows"],
                "sha256": observation["sha256"],
            }
        )
    return {
        "schema_version": 2,
        "completion_status": "complete",
        "start_time_utc": start_time,
        "finish_time_utc": finish_time,
        "elapsed_seconds": elapsed_seconds,
        "code_commit": code_commit,
        "command_arguments": command_arguments,
        "path_bases": {
            "repository_root": "repository root",
            "preparation_manifest_directory": "directory containing this manifest",
        },
        "selection_manifest": {
            "path": repository_relative_path(selection_manifest, repository_root),
            "path_base": "repository_root",
            "sha256": sha256_file(selection_manifest),
            "selection_rule_name": selection_document["selection_rule_name"],
            "repeated_hash_list_sha256": selection_document["repeated_hash_list_sha256"],
        },
        "inputs": input_records,
        "rows": row_count,
        "feature_count": feature_count,
        "extractor": {
            "version": selection_document["extractor_version"],
            "dimension": selection_document["extractor_dimension"],
        },
        "dependency_versions": versions,
        "feature_dtype": "float32",
        "label_dtype": "int32",
        "artifacts": {
            "features": {
                "path": feature_path.name,
                "path_base": "preparation_manifest_directory",
                "sha256": sha256_file(feature_path),
                "size_bytes": feature_path.stat().st_size,
                **feature_validation,
            },
            "labels": {
                "path": label_path.name,
                "path_base": "preparation_manifest_directory",
                "sha256": sha256_file(label_path),
                "size_bytes": label_path.stat().st_size,
                **label_validation,
            },
            "metadata": {
                "path": metadata_path.name,
                "path_base": "preparation_manifest_directory",
                "sha256": sha256_file(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
                **metadata_validation,
            },
        },
    }


def execute_preparation(
    repository_root: Path,
    selection_manifest: Path,
    selection_document: dict[str, object],
    inputs: list[tuple[str, Path]],
    metadata: pd.DataFrame,
    input_observations: list[dict[str, object]],
    output_dir: Path,
    vectorize_function: Callable[..., object],
    extractor: object,
    metadata_validator: Callable[[pd.DataFrame], object],
    overwrite: bool = False,
    expected_rows: int = EXPECTED_PREPARATION_ROWS,
    expected_features: int = EXPECTED_FEATURE_COUNT,
    versions: dict[str, str] | None = None,
    commit_identifier: str | None = None,
) -> Path:
    repository_relative_path(output_dir, repository_root)
    if output_dir.exists() and not overwrite:
        raise ValueError("completed preparation already exists; pass --overwrite to replace it")
    if len(metadata) != expected_rows:
        raise ValueError(f"metadata row count mismatch: expected {expected_rows}, found {len(metadata)}")
    if list(metadata.columns) != METADATA_COLUMNS:
        raise ValueError(f"metadata columns must be exactly {METADATA_COLUMNS}")
    metadata_validator(metadata)

    start_time = utc_now()
    start_clock = time.monotonic()
    staging_dir = create_preparation_staging_directory(output_dir)
    final_manifest = output_dir / "preparation_manifest.json"
    try:
        feature_path = staging_dir / "X_test.dat"
        label_path = staging_dir / "y_test.dat"
        metadata_path = staging_dir / "test_metadata.parquet"
        vectorize_function(
            feature_path,
            label_path,
            [path for _, path in inputs],
            extractor,
            expected_rows,
            "label",
        )
        feature_validation = validate_feature_file(
            feature_path,
            expected_rows=expected_rows,
            expected_features=expected_features,
        )
        expected_labels = metadata["label"].to_numpy(dtype=np.int32)
        validated_labels = validate_vectorized_label_file(
            label_path,
            expected_labels,
            expected_rows=expected_rows,
        )
        label_validation = {
            "count": expected_rows,
            "dtype": "int32",
            "allowed_values": True,
            "aligned_with_metadata": True,
        }
        metadata.to_parquet(metadata_path, index=False)
        observed_metadata, metadata_validation = validate_metadata_file(
            metadata_path,
            metadata,
            metadata_validator,
            expected_rows=expected_rows,
        )
        if not np.array_equal(
            validated_labels,
            observed_metadata["label"].to_numpy(dtype=np.int32),
        ):
            raise ValueError("metadata parquet labels do not match the validated label file")
        metadata_validation["labels_aligned"] = True
        finish_time = utc_now()
        elapsed_seconds = round(time.monotonic() - start_clock, 6)
        command_arguments = {
            "selection_manifest": repository_relative_path(selection_manifest, repository_root),
            "output_dir": repository_relative_path(output_dir, repository_root),
            "execute_vectorization": True,
            "overwrite": overwrite,
        }
        provenance_document = build_preparation_provenance(
            repository_root,
            selection_manifest,
            selection_document,
            input_observations,
            expected_rows,
            expected_features,
            feature_path,
            label_path,
            metadata_path,
            feature_validation,
            label_validation,
            metadata_validation,
            start_time,
            finish_time,
            elapsed_seconds,
            commit_identifier or current_commit(repository_root),
            command_arguments,
            versions or dependency_versions(),
        )
        staged_manifest = staging_dir / "preparation_manifest.json"
        staged_manifest.write_text(
            json.dumps(provenance_document, indent=2) + "\n",
            encoding="utf-8",
        )
        load_prepared_artifacts(
            staged_manifest,
            repository_root,
            expected_rows=expected_rows,
            expected_features=expected_features,
            metadata_validator=metadata_validator,
        )
        publish_prepared_directory(staging_dir, output_dir, overwrite)
        return final_manifest
    except Exception:
        remove_preparation_operation_directory(staging_dir, output_dir, "staging")
        raise


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from thrember import PEFeatureExtractor

    repository_root = Path(__file__).resolve().parents[1]
    repository_relative_path(args.selection_manifest, repository_root)
    repository_relative_path(args.output_dir, repository_root)
    extractor = PEFeatureExtractor()
    selection_document, manifest = load_completed_selection_manifest(
        args.selection_manifest,
        repository_root,
        extractor,
    )
    print("selected input manifest:")
    all_rows = []
    input_observations = []
    observed_counts = Counter()
    for file_type, path in manifest:
        count, rows = inspect_jsonl(path)
        checksum = sha256_file(path)
        print(f"- {file_type}: {path.resolve()} ({count} rows, sha256={checksum})")
        observed_counts[file_type] += count
        all_rows.extend(rows)
        input_observations.append(
            {
                "file_type": file_type,
                "path": path,
                "rows": count,
                "sha256": checksum,
            }
        )
    validate_aggregate_record_counts(observed_counts)
    metadata = pd.DataFrame(all_rows)
    report = validate_reviewed_selection_metadata(metadata)
    print(report.as_text())
    if not args.execute_vectorization:
        print("validation complete; vectorization was not requested")
        return
    from thrember.model import vectorize_subset

    preparation_manifest_path = execute_preparation(
        repository_root,
        args.selection_manifest,
        selection_document,
        manifest,
        metadata,
        input_observations,
        args.output_dir,
        vectorize_subset,
        extractor,
        validate_reviewed_selection_metadata,
        overwrite=args.overwrite,
    )
    print(
        f"prepared {len(metadata)} aligned rows with {EXPECTED_FEATURE_COUNT} features: "
        f"{repository_relative_path(preparation_manifest_path, repository_root)}"
    )


if __name__ == "__main__":
    main()
