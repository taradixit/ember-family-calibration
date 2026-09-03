#!/usr/bin/env python3
"""Validate reviewed inputs and write aggregate and family summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.data_validation import validate_reviewed_selection_inputs  # noqa: E402
from ember_calibration.family_analysis import family_metrics  # noqa: E402
from ember_calibration.inference_artifacts import load_inference_artifacts  # noqa: E402
from ember_calibration.metrics import (  # noqa: E402
    CALIBRATION_THRESHOLD,
    accuracy,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    roc_auc,
)
from ember_calibration.selection import repository_relative_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--minimum-family-count", type=int, default=100)
    return parser.parse_args(argv)


def write_analysis_outputs(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    output_dir: Path,
    threshold: float,
    bins: int,
    minimum_family_count: int,
) -> None:
    report = validate_reviewed_selection_inputs(metadata, predictions)
    print(report.as_text())
    labels = metadata["label"].to_numpy()
    summary = {
        "accuracy": accuracy(labels, predictions, threshold),
        "roc_auc": roc_auc(labels, predictions),
        "brier_score": brier_score(labels, predictions),
        "ece": expected_calibration_error(labels, predictions, bins),
        "mce": maximum_calibration_error(labels, predictions, bins),
        "decision_threshold": threshold,
        "calibration_decision_threshold": CALIBRATION_THRESHOLD,
        "calibration_confidence_range": [0.5, 1.0],
        "calibration_bins": bins,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(reliability_bins(labels, predictions, bins)).to_csv(
        output_dir / "reliability_bins.csv", index=False
    )
    family_metrics(metadata, predictions, minimum_family_count, threshold, bins).to_csv(
        output_dir / "family_metrics.csv", index=False
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    repository_relative_path(args.inference_manifest, repository_root)
    repository_relative_path(args.output_dir, repository_root)
    inference = load_inference_artifacts(args.inference_manifest, repository_root)
    metadata = pd.read_parquet(inference["metadata"])
    predictions = np.load(inference["predictions"], mmap_mode="r", allow_pickle=False)
    try:
        write_analysis_outputs(
            metadata,
            predictions,
            args.output_dir,
            args.threshold,
            args.bins,
            args.minimum_family_count,
        )
    finally:
        del predictions


if __name__ == "__main__":
    main()
