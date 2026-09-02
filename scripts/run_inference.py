#!/usr/bin/env python3
"""Validate inputs and optionally run the released LightGBM detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.archive_manifest import sha256_file  # noqa: E402
from ember_calibration.data_validation import (  # noqa: E402
    validate_reviewed_selection_inputs,
    validate_reviewed_selection_metadata,
)
from ember_calibration.prepared_artifacts import load_prepared_artifacts  # noqa: E402
from ember_calibration.selection import repository_relative_path  # noqa: E402
from ember_calibration.upstream import HISTORICAL_MODEL_SHA256, MODEL_REVISION  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def preparation_relative_path(path: Path, preparation_manifest: Path) -> str:
    try:
        return path.resolve().relative_to(preparation_manifest.parent.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("prepared artifact is outside the preparation directory") from error


def build_prediction_provenance(
    repository_root: Path,
    model_path: Path,
    model_checksum: str,
    preparation_manifest: Path,
    prepared: dict[str, object],
    lightgbm_version: str,
    model_feature_count: int,
) -> dict[str, object]:
    features = Path(prepared["features"])
    metadata = Path(prepared["metadata"])
    repository_relative_path(features, repository_root)
    repository_relative_path(metadata, repository_root)
    return {
        "path_bases": {
            "repository_root": "repository root",
            "preparation_manifest_directory": "directory containing the preparation manifest",
        },
        "model": {
            "path": repository_relative_path(model_path, repository_root),
            "path_base": "repository_root",
            "sha256": model_checksum,
            "repository_revision": MODEL_REVISION,
        },
        "lightgbm_version": lightgbm_version,
        "model_feature_count": model_feature_count,
        "preparation_manifest": {
            "path": repository_relative_path(preparation_manifest, repository_root),
            "path_base": "repository_root",
            "sha256": sha256_file(preparation_manifest),
        },
        "features": {
            "path": preparation_relative_path(features, preparation_manifest),
            "path_base": "preparation_manifest_directory",
            "sha256": sha256_file(features),
        },
        "metadata": {
            "path": preparation_relative_path(metadata, preparation_manifest),
            "path_base": "preparation_manifest_directory",
            "sha256": sha256_file(metadata),
        },
        "rows": int(prepared["rows"]),
        "prepared_feature_count": int(prepared["feature_count"]),
    }


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    repository_relative_path(args.model, repository_root)
    repository_relative_path(args.preparation_manifest, repository_root)
    repository_relative_path(args.output, repository_root)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    prepared = load_prepared_artifacts(args.preparation_manifest)
    model_checksum = sha256_file(args.model)
    if model_checksum != HISTORICAL_MODEL_SHA256:
        raise ValueError(f"unexpected model checksum: {model_checksum}")
    metadata = pd.read_parquet(prepared["metadata"])
    if len(metadata) != prepared["rows"]:
        raise ValueError("metadata row count does not match preparation manifest")
    report = validate_reviewed_selection_metadata(metadata)
    print(report.as_text())
    print(f"model sha256: {model_checksum}")
    print(f"feature file sha256: {sha256_file(prepared['features'])}")
    if not args.execute:
        print("validation complete; inference was not requested")
        return
    import lightgbm as lgb

    rows = int(prepared["rows"])
    feature_count = int(prepared["feature_count"])
    features = np.memmap(prepared["features"], dtype=np.float32, mode="r", shape=(rows, feature_count))
    model = lgb.Booster(model_file=str(args.model))
    model_feature_count = model.num_feature()
    if model_feature_count != feature_count:
        raise ValueError(
            f"model/preparation feature-count mismatch: {model_feature_count} != {feature_count}"
        )
    predictions = model.predict(features)
    validate_reviewed_selection_inputs(metadata, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, predictions)
    provenance = build_prediction_provenance(
        repository_root,
        args.model,
        model_checksum,
        args.preparation_manifest,
        prepared,
        lgb.__version__,
        model_feature_count,
    )
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
