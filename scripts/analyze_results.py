#!/usr/bin/env python3
"""Validate corrected inputs and write aggregate and family summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.data_validation import validate_inputs  # noqa: E402
from ember_calibration.family_analysis import family_metrics  # noqa: E402
from ember_calibration.metrics import (  # noqa: E402
    CALIBRATION_THRESHOLD,
    accuracy,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    roc_auc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--minimum-family-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_parquet(args.metadata)
    predictions = np.load(args.predictions, allow_pickle=False)
    report = validate_inputs(metadata, predictions)
    print(report.as_text())
    labels = metadata["label"].to_numpy()
    summary = {
        "accuracy": accuracy(labels, predictions, args.threshold),
        "roc_auc": roc_auc(labels, predictions),
        "brier_score": brier_score(labels, predictions),
        "ece": expected_calibration_error(labels, predictions, args.bins),
        "mce": maximum_calibration_error(labels, predictions, args.bins),
        "decision_threshold": args.threshold,
        "calibration_decision_threshold": CALIBRATION_THRESHOLD,
        "calibration_confidence_range": [0.5, 1.0],
        "calibration_bins": args.bins,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(reliability_bins(labels, predictions, args.bins)).to_csv(
        args.output_dir / "reliability_bins.csv", index=False
    )
    family_metrics(metadata, predictions, args.minimum_family_count, args.threshold, args.bins).to_csv(
        args.output_dir / "family_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
