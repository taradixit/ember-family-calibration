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
from ember_calibration.data_validation import validate_metadata, validate_predictions  # noqa: E402
from ember_calibration.prepared_artifacts import load_prepared_artifacts  # noqa: E402
from ember_calibration.upstream import HISTORICAL_MODEL_SHA256, MODEL_REVISION  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    prepared = load_prepared_artifacts(args.preparation_manifest)
    model_checksum = sha256_file(args.model)
    if model_checksum != HISTORICAL_MODEL_SHA256:
        raise ValueError(f"unexpected model checksum: {model_checksum}")
    metadata = pd.read_parquet(prepared["metadata"])
    if len(metadata) != prepared["rows"]:
        raise ValueError("metadata row count does not match preparation manifest")
    report = validate_metadata(metadata)
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
    validate_predictions(predictions, expected_length=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, predictions)
    provenance = {
        "model": str(args.model.resolve()),
        "model_sha256": model_checksum,
        "model_repository_revision": MODEL_REVISION,
        "lightgbm_version": lgb.__version__,
        "model_feature_count": model_feature_count,
        "preparation_manifest": str(args.preparation_manifest.resolve()),
        "preparation_manifest_sha256": sha256_file(args.preparation_manifest),
        "features": str(prepared["features"]),
        "features_sha256": sha256_file(prepared["features"]),
        "metadata": str(prepared["metadata"]),
        "metadata_sha256": sha256_file(prepared["metadata"]),
        "rows": rows,
        "prepared_feature_count": feature_count,
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
