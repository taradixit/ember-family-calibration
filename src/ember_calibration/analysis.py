"""Aggregate, family, sensitivity, and plotting helpers for corrected analysis."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from .archive_manifest import sha256_file
from .data_validation import validate_reviewed_selection_inputs
from .family_analysis import family_metrics
from .metrics import (
    CALIBRATION_THRESHOLD,
    accuracy,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    roc_auc,
)

CALIBRATION_DEFINITION = (
    "predicted class p >= 0.5; predicted-label confidence max(p, 1-p); "
    "equal-width [0.5, 1.0] bins; left-closed/right-open except final includes 1.0"
)
EXPECTED_OUTPUTS = (
    "aggregate_metrics.json",
    "reliability_bins.csv",
    "family_metrics.csv",
    "calibration_bin_sensitivity.csv",
    "family_minimum_sensitivity.csv",
    "reliability_diagram.png",
    "family_failures.png",
)


def validate_analysis_parameters(
    threshold: float,
    primary_bins: int,
    primary_family_minimum: int,
    sensitivity_bins: Iterable[int],
    sensitivity_family_minimums: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    if not isinstance(primary_bins, int) or isinstance(primary_bins, bool) or primary_bins < 1:
        raise ValueError("primary bin count must be a positive integer")
    if (
        not isinstance(primary_family_minimum, int)
        or isinstance(primary_family_minimum, bool)
        or primary_family_minimum < 1
    ):
        raise ValueError("primary family minimum must be a positive integer")
    bin_values = tuple(sensitivity_bins)
    minimum_values = tuple(sensitivity_family_minimums)
    if (
        not bin_values
        or len(set(bin_values)) != len(bin_values)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in bin_values)
    ):
        raise ValueError("sensitivity bin counts must be unique positive integers")
    if (
        not minimum_values
        or len(set(minimum_values)) != len(minimum_values)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in minimum_values
        )
    ):
        raise ValueError("sensitivity family minimums must be unique positive integers")
    if primary_bins not in bin_values:
        raise ValueError("sensitivity bin counts must include the primary bin count")
    if primary_family_minimum not in minimum_values:
        raise ValueError("sensitivity family minimums must include the primary family minimum")
    return bin_values, minimum_values


def usable_malicious_families(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    malicious = metadata[metadata["label"] == 1].copy()
    normalized = malicious["family"].astype("string").str.strip().str.casefold()
    unusable = normalized.isna() | normalized.isin({"", "unknown", "none", "nan"})
    return malicious.loc[~unusable].copy(), malicious.loc[unusable, "family"]


def family_exclusion_counts(metadata: pd.DataFrame, minimum_count: int) -> dict[str, int]:
    usable, unusable = usable_malicious_families(metadata)
    counts = usable["family"].value_counts(dropna=False)
    eligible = counts[counts >= minimum_count]
    below = counts[counts < minimum_count]
    return {
        "malicious_rows": int((metadata["label"] == 1).sum()),
        "unusable_family_rows": int(len(unusable)),
        "usable_family_rows": int(len(usable)),
        "below_minimum_family_count": int(len(below)),
        "below_minimum_rows": int(below.sum()),
        "eligible_family_count": int(len(eligible)),
        "eligible_malicious_rows": int(eligible.sum()),
    }


def _top_family_names(frame: pd.DataFrame, metric: str, limit: int = 10) -> list[str]:
    ordered = frame.sort_values([metric, "family"], ascending=[False, True], kind="stable")
    return ordered.head(limit)["family"].astype(str).tolist()


def _overlap(primary: list[str], comparison: list[str]) -> tuple[int, str]:
    shared = sorted(set(primary) & set(comparison))
    return len(shared), ";".join(shared)


def calibration_sensitivity(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    family_table: pd.DataFrame,
    bin_counts: Iterable[int],
    threshold: float,
    primary_bins: int,
    primary_family_minimum: int,
) -> pd.DataFrame:
    labels = metadata["label"].to_numpy()
    primary_ece_table = family_metrics(
        metadata,
        predictions,
        minimum_count=primary_family_minimum,
        threshold=threshold,
        n_bins=primary_bins,
    )
    primary_top = _top_family_names(primary_ece_table, "ece")
    rows = []
    for bin_count in bin_counts:
        comparison = family_metrics(
            metadata,
            predictions,
            minimum_count=primary_family_minimum,
            threshold=threshold,
            n_bins=bin_count,
        )
        overlap_count, overlap_names = _overlap(
            primary_top,
            _top_family_names(comparison, "ece"),
        )
        rows.append(
            {
                "bin_count": bin_count,
                "aggregate_ece": expected_calibration_error(labels, predictions, bin_count),
                "aggregate_mce": maximum_calibration_error(labels, predictions, bin_count),
                "primary_top_10_family_ece_overlap_count": overlap_count,
                "primary_top_10_family_ece_overlap_families": overlap_names,
            }
        )
    result = pd.DataFrame(rows)
    if primary_bins in result["bin_count"].values:
        primary_row = result[result["bin_count"] == primary_bins].iloc[0]
        if primary_row["primary_top_10_family_ece_overlap_count"] != min(10, len(family_table)):
            raise ValueError("primary family-ECE overlap is inconsistent")
    return result


def family_minimum_sensitivity(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    minimums: Iterable[int],
    threshold: float,
    n_bins: int,
    primary_minimum: int,
) -> pd.DataFrame:
    primary = family_metrics(
        metadata,
        predictions,
        minimum_count=primary_minimum,
        threshold=threshold,
        n_bins=n_bins,
    )
    primary_fnr = _top_family_names(primary, "false_negative_rate")
    primary_ece = _top_family_names(primary, "ece")
    rows = []
    for minimum in minimums:
        comparison = family_metrics(
            metadata,
            predictions,
            minimum_count=minimum,
            threshold=threshold,
            n_bins=n_bins,
        )
        fnr_count, fnr_names = _overlap(
            primary_fnr,
            _top_family_names(comparison, "false_negative_rate"),
        )
        ece_count, ece_names = _overlap(primary_ece, _top_family_names(comparison, "ece"))
        rows.append(
            {
                "minimum_malicious_family_count": minimum,
                "eligible_family_count": int(len(comparison)),
                "eligible_malicious_row_count": int(comparison["count"].sum()),
                "primary_top_10_false_negative_rate_overlap_count": fnr_count,
                "primary_top_10_false_negative_rate_overlap_families": fnr_names,
                "primary_top_10_ece_overlap_count": ece_count,
                "primary_top_10_ece_overlap_families": ece_names,
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(reliability: pd.DataFrame, output_path: Path) -> None:
    nonempty = reliability[reliability["count"] > 0]
    figure, (calibration_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.2),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    calibration_axis.plot([0.5, 1.0], [0.5, 1.0], "--", color="0.45", label="Perfect calibration")
    calibration_axis.plot(
        nonempty["mean_confidence"],
        nonempty["accuracy"],
        marker="o",
        color="#2B6F8E",
        label="Observed",
    )
    calibration_axis.set(xlim=(0.5, 1.0), ylim=(0.5, 1.0), xlabel="Mean confidence", ylabel="Observed accuracy")
    calibration_axis.legend(frameon=False)
    widths = reliability["upper"] - reliability["lower"]
    count_axis.bar(reliability["lower"], reliability["count"], width=widths, align="edge", color="#7AA6B8")
    count_axis.set(xlim=(0.5, 1.0), xlabel="Predicted-label confidence", ylabel="Count")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_family_failures(families: pd.DataFrame, output_path: Path) -> None:
    selected = families.sort_values(
        ["false_negative_rate", "ece", "family"],
        ascending=[False, False, True],
        kind="stable",
    ).head(15)
    selected = selected.iloc[::-1]
    labels = [f"{family} (n={count})" for family, count in zip(selected["family"], selected["count"], strict=True)]
    figure, (fnr_axis, ece_axis) = plt.subplots(1, 2, figsize=(13, 7), sharey=True, constrained_layout=True)
    positions = np.arange(len(selected))
    fnr_axis.barh(positions, selected["false_negative_rate"], color="#A44A3F")
    fnr_axis.set(yticks=positions, yticklabels=labels, xlabel="False-negative rate", title="Highest family false-negative rates")
    ece_axis.barh(positions, selected["ece"], color="#3D7A6C")
    ece_axis.set(xlabel="15-bin ECE", title="Predicted-label confidence calibration")
    fnr_axis.set_xlim(left=0)
    ece_axis.set_xlim(left=0)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_analysis_outputs(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    output_dir: Path,
    threshold: float,
    primary_bins: int,
    primary_family_minimum: int,
    sensitivity_bins: Iterable[int],
    sensitivity_family_minimums: Iterable[int],
    provenance: dict[str, object],
) -> dict[str, object]:
    bin_values, minimum_values = validate_analysis_parameters(
        threshold,
        primary_bins,
        primary_family_minimum,
        sensitivity_bins,
        sensitivity_family_minimums,
    )
    report = validate_reviewed_selection_inputs(metadata, predictions)
    labels = metadata["label"].to_numpy()
    aggregate = {
        "accuracy": accuracy(labels, predictions, threshold),
        "roc_auc": roc_auc(labels, predictions),
        "brier_score": brier_score(labels, predictions),
        "ece": expected_calibration_error(labels, predictions, primary_bins),
        "mce": maximum_calibration_error(labels, predictions, primary_bins),
        "decision_threshold": threshold,
        "calibration_decision_threshold": CALIBRATION_THRESHOLD,
        "calibration_confidence_range": [0.5, 1.0],
        "calibration_bins": primary_bins,
    }
    reliability = pd.DataFrame(reliability_bins(labels, predictions, primary_bins))
    families = family_metrics(
        metadata,
        predictions,
        minimum_count=primary_family_minimum,
        threshold=threshold,
        n_bins=primary_bins,
    )
    bin_sensitivity = calibration_sensitivity(
        metadata,
        predictions,
        families,
        bin_values,
        threshold,
        primary_bins,
        primary_family_minimum,
    )
    minimum_sensitivity = family_minimum_sensitivity(
        metadata,
        predictions,
        minimum_values,
        threshold,
        primary_bins,
        primary_family_minimum,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    reliability.to_csv(output_dir / "reliability_bins.csv", index=False)
    families.to_csv(output_dir / "family_metrics.csv", index=False)
    bin_sensitivity.to_csv(output_dir / "calibration_bin_sensitivity.csv", index=False)
    minimum_sensitivity.to_csv(output_dir / "family_minimum_sensitivity.csv", index=False)
    plot_reliability(reliability, output_dir / "reliability_diagram.png")
    plot_family_failures(families, output_dir / "family_failures.png")
    outputs = {
        filename: {
            "size_bytes": (output_dir / filename).stat().st_size,
            "sha256": sha256_file(output_dir / filename),
        }
        for filename in EXPECTED_OUTPUTS
    }
    manifest = {
        "schema_version": 1,
        "completion_status": "complete",
        **provenance,
        "row_count": report.row_count,
        "prediction_shape": [len(predictions)],
        "prediction_dtype": str(np.asarray(predictions).dtype),
        "decision_threshold": threshold,
        "calibration_definition": CALIBRATION_DEFINITION,
        "primary_bin_count": primary_bins,
        "sensitivity_bin_counts": list(bin_values),
        "primary_family_minimum": primary_family_minimum,
        "sensitivity_family_minimums": list(minimum_values),
        "family_exclusion_counts": family_exclusion_counts(metadata, primary_family_minimum),
        "output_filenames": [*EXPECTED_OUTPUTS, "analysis_manifest.json"],
        "outputs": outputs,
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
