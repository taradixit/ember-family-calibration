#!/usr/bin/env python3
"""Write corrected aggregate, family, sensitivity, and plotting outputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.analysis import write_analysis_outputs  # noqa: E402
from ember_calibration.archive_manifest import sha256_file  # noqa: E402
from ember_calibration.inference_artifacts import load_inference_artifacts  # noqa: E402
from ember_calibration.selection import repository_relative_path  # noqa: E402


def positive_integer_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not parsed or len(set(parsed)) != len(parsed) or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected unique comma-separated positive integers")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--minimum-family-count", type=int, default=100)
    parser.add_argument(
        "--sensitivity-bins",
        type=positive_integer_list,
        default=(10, 15, 20, 30),
    )
    parser.add_argument(
        "--sensitivity-family-minimums",
        type=positive_integer_list,
        default=(50, 100, 200),
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_commit(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    repository_relative_path(args.inference_manifest, repository_root)
    repository_relative_path(args.output_dir, repository_root)
    inference = load_inference_artifacts(args.inference_manifest, repository_root)
    inference_document = json.loads(args.inference_manifest.read_text(encoding="utf-8"))
    metadata = pd.read_parquet(inference["metadata"])
    predictions = np.load(inference["predictions"], mmap_mode="r", allow_pickle=False)
    try:
        provenance = {
            "execution_time_utc": utc_now(),
            "implementation_commit": current_commit(repository_root),
            "input_paths": {
                "inference_manifest": repository_relative_path(
                    args.inference_manifest, repository_root
                ),
                "predictions": repository_relative_path(inference["predictions"], repository_root),
                "preparation_manifest": repository_relative_path(
                    inference["preparation_manifest"], repository_root
                ),
                "metadata": repository_relative_path(inference["metadata"], repository_root),
                "model": repository_relative_path(inference["model"], repository_root),
            },
            "input_sha256": {
                "inference_manifest": sha256_file(args.inference_manifest),
                "predictions": sha256_file(inference["predictions"]),
                "preparation_manifest": sha256_file(inference["preparation_manifest"]),
                "metadata": sha256_file(inference["metadata"]),
                "model": sha256_file(inference["model"]),
            },
            "model_revision": inference_document["model"]["repository_revision"],
            "dependency_versions": {
                name: importlib.metadata.version(name)
                for name in ("matplotlib", "numpy", "pandas", "scikit-learn")
            },
        }
        write_analysis_outputs(
            metadata,
            predictions,
            args.output_dir,
            args.threshold,
            args.bins,
            args.minimum_family_count,
            args.sensitivity_bins,
            args.sensitivity_family_minimums,
            provenance,
        )
    finally:
        del predictions


if __name__ == "__main__":
    main()
