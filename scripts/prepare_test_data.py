#!/usr/bin/env python3
"""Validate a completed reviewed selection before vectorization."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.archive_manifest import sha256_file  # noqa: E402
from ember_calibration.data_validation import validate_reviewed_selection_metadata  # noqa: E402
from ember_calibration.preparation import validate_vectorized_label_file  # noqa: E402
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
    return parser.parse_args(argv)


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
        "schema_version": 1,
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
        "feature_dtype": "float32",
        "label_dtype": "int32",
        "artifacts": {
            "features": {
                "path": feature_path.name,
                "path_base": "preparation_manifest_directory",
                "sha256": sha256_file(feature_path),
                "size_bytes": feature_path.stat().st_size,
            },
            "labels": {
                "path": label_path.name,
                "path_base": "preparation_manifest_directory",
                "sha256": sha256_file(label_path),
                "size_bytes": label_path.stat().st_size,
            },
            "metadata": {
                "path": metadata_path.name,
                "path_base": "preparation_manifest_directory",
                "sha256": sha256_file(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
            },
        },
        "alignment_check": "vectorized labels exactly match metadata labels",
    }


def main() -> None:
    args = parse_args()
    from thrember import PEFeatureExtractor

    repository_root = Path(__file__).resolve().parents[1]
    repository_relative_path(args.selection_manifest, repository_root)
    repository_relative_path(args.output_dir, repository_root)
    selection_document, manifest = load_completed_selection_manifest(
        args.selection_manifest,
        repository_root,
        PEFeatureExtractor(),
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "X_test.dat"
    label_path = args.output_dir / "y_test.dat"
    paths = [path for _, path in manifest]
    vectorize_subset(feature_path, label_path, paths, PEFeatureExtractor(), len(metadata), "label")
    validate_vectorized_label_file(label_path, metadata["label"].to_numpy(dtype=np.int32))
    feature_bytes_per_row, remainder = divmod(feature_path.stat().st_size, len(metadata))
    if remainder or feature_bytes_per_row % np.dtype(np.float32).itemsize:
        raise RuntimeError("feature file size is not consistent with the metadata row count")
    feature_count = feature_bytes_per_row // np.dtype(np.float32).itemsize
    metadata_path = args.output_dir / "test_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    preparation_manifest_path = args.output_dir / "preparation_manifest.json"
    provenance_document = build_preparation_provenance(
        repository_root,
        args.selection_manifest,
        selection_document,
        input_observations,
        len(metadata),
        feature_count,
        feature_path,
        label_path,
        metadata_path,
    )
    preparation_manifest_path.write_text(json.dumps(provenance_document, indent=2) + "\n")
    print(f"prepared {len(metadata)} aligned rows with {feature_count} features")


if __name__ == "__main__":
    main()
