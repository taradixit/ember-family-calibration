#!/usr/bin/env python3
"""Validate inputs and optionally run the released LightGBM detector in batches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.archive_manifest import sha256_file  # noqa: E402
from ember_calibration.data_validation import validate_reviewed_selection_metadata  # noqa: E402
from ember_calibration.inference import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    create_inference_staging_directory,
    positive_batch_size,
    publish_inference_directory,
    remove_inference_operation_directory,
    write_batched_predictions,
)
from ember_calibration.inference_artifacts import (  # noqa: E402
    PREDICTION_MEANING,
    load_inference_artifacts,
)
from ember_calibration.prepared_artifacts import load_prepared_artifacts  # noqa: E402
from ember_calibration.selection import repository_relative_path  # noqa: E402
from ember_calibration.upstream import HISTORICAL_MODEL_SHA256, MODEL_REVISION  # noqa: E402


def parse_batch_size(value: str) -> int:
    try:
        return positive_batch_size(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=parse_batch_size, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.overwrite and not args.execute:
        parser.error("--overwrite requires --execute")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_commit(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def build_inference_manifest(
    repository_root: Path,
    model_path: Path,
    model_checksum: str,
    preparation_manifest: Path,
    prepared: dict[str, object],
    output_dir: Path,
    prediction_path: Path,
    prediction_validation: dict[str, object],
    lightgbm_version: str,
    model_feature_count: int,
    batch_size: int,
    overwrite: bool,
    start_time: str,
    finish_time: str,
    elapsed_seconds: float,
    code_commit: str,
) -> dict[str, object]:
    repository_relative_path(output_dir, repository_root)
    preparation_document = json.loads(preparation_manifest.read_text(encoding="utf-8"))
    preparation_checksums = {
        name: preparation_document["artifacts"][name]["sha256"]
        for name in ("features", "labels", "metadata")
    }
    return {
        "schema_version": 1,
        "completion_status": "complete",
        "start_time_utc": start_time,
        "finish_time_utc": finish_time,
        "elapsed_seconds": elapsed_seconds,
        "code_commit": code_commit,
        "command_arguments": {
            "model": repository_relative_path(model_path, repository_root),
            "preparation_manifest": repository_relative_path(
                preparation_manifest, repository_root
            ),
            "output_dir": repository_relative_path(output_dir, repository_root),
            "batch_size": batch_size,
            "execute": True,
            "overwrite": overwrite,
        },
        "path_bases": {
            "repository_root": "repository root",
            "inference_manifest_directory": "directory containing this manifest",
        },
        "batch_size": batch_size,
        "prediction_batch_count": prediction_validation["batch_count"],
        "rows": int(prepared["rows"]),
        "feature_count": int(prepared["feature_count"]),
        "prediction_meaning": PREDICTION_MEANING,
        "threshold_applied": False,
        "model": {
            "repository_revision": MODEL_REVISION,
            "path": repository_relative_path(model_path, repository_root),
            "path_base": "repository_root",
            "sha256": model_checksum,
        },
        "lightgbm_version": lightgbm_version,
        "model_feature_count": model_feature_count,
        "preparation_manifest": {
            "path": repository_relative_path(preparation_manifest, repository_root),
            "path_base": "repository_root",
            "sha256": sha256_file(preparation_manifest),
        },
        "preparation_artifact_sha256": preparation_checksums,
        "prediction": {
            "filename": "predictions.npy",
            "path": prediction_path.name,
            "path_base": "inference_manifest_directory",
            "dtype": prediction_validation["dtype"],
            "shape": prediction_validation["shape"],
            "size_bytes": prediction_validation["size_bytes"],
            "sha256": sha256_file(prediction_path),
            "minimum": prediction_validation["minimum"],
            "maximum": prediction_validation["maximum"],
            "finite_values": prediction_validation["finite_values"],
            "range_valid": prediction_validation["range_valid"],
            "exact_row_count": prediction_validation["exact_row_count"],
        },
    }


def execute_inference(
    repository_root: Path,
    model_path: Path,
    model_checksum: str,
    preparation_manifest: Path,
    prepared: dict[str, object],
    output_dir: Path,
    model: object,
    lightgbm_version: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
    commit_identifier: str | None = None,
    manifest_validator: Callable[[Path], object] | None = None,
) -> Path:
    repository_relative_path(output_dir, repository_root)
    positive_batch_size(batch_size)
    if output_dir.exists() and not overwrite:
        raise ValueError("completed inference already exists; pass --overwrite to replace it")
    rows = int(prepared["rows"])
    feature_count = int(prepared["feature_count"])
    model_feature_count = int(model.num_feature())
    if model_feature_count != feature_count:
        raise ValueError(
            f"model/preparation feature-count mismatch: {model_feature_count} != {feature_count}"
        )

    start_time = utc_now()
    start_clock = time.monotonic()
    staging_dir = create_inference_staging_directory(output_dir)
    final_manifest = output_dir / "inference_manifest.json"
    try:
        prediction_path = staging_dir / "predictions.npy"
        prediction_validation = write_batched_predictions(
            Path(prepared["features"]),
            prediction_path,
            model,
            rows,
            feature_count,
            batch_size,
        )
        finish_time = utc_now()
        elapsed_seconds = round(time.monotonic() - start_clock, 6)
        manifest_document = build_inference_manifest(
            repository_root,
            model_path,
            model_checksum,
            preparation_manifest,
            prepared,
            output_dir,
            prediction_path,
            prediction_validation,
            lightgbm_version,
            model_feature_count,
            batch_size,
            overwrite,
            start_time,
            finish_time,
            elapsed_seconds,
            commit_identifier or current_commit(repository_root),
        )
        staged_manifest = staging_dir / "inference_manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest_document, indent=2) + "\n",
            encoding="utf-8",
        )
        if manifest_validator is None:
            load_inference_artifacts(
                staged_manifest,
                repository_root,
                expected_rows=rows,
                expected_features=feature_count,
                expected_model_sha256=model_checksum,
            )
        else:
            manifest_validator(staged_manifest)
        publish_inference_directory(staging_dir, output_dir, overwrite)
        return final_manifest
    except Exception:
        remove_inference_operation_directory(staging_dir, output_dir, "staging")
        raise


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    repository_relative_path(args.model, repository_root)
    repository_relative_path(args.preparation_manifest, repository_root)
    repository_relative_path(args.output_dir, repository_root)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    prepared = load_prepared_artifacts(args.preparation_manifest, repository_root)
    model_checksum = sha256_file(args.model)
    if model_checksum != HISTORICAL_MODEL_SHA256:
        raise ValueError(f"unexpected model checksum: {model_checksum}")
    metadata = pd.read_parquet(prepared["metadata"])
    report = validate_reviewed_selection_metadata(metadata)
    print(report.as_text())
    print(f"model sha256: {model_checksum}")
    print(f"feature file sha256: {sha256_file(prepared['features'])}")
    if not args.execute:
        print("validation complete; inference was not requested")
        return

    import lightgbm as lgb

    model = lgb.Booster(model_file=str(args.model))
    manifest_path = execute_inference(
        repository_root,
        args.model,
        model_checksum,
        args.preparation_manifest,
        prepared,
        args.output_dir,
        model,
        lgb.__version__,
        args.batch_size,
        overwrite=args.overwrite,
    )
    print(
        f"wrote {prepared['rows']} ordered probabilities in bounded batches: "
        f"{repository_relative_path(manifest_path, repository_root)}"
    )


if __name__ == "__main__":
    main()
