import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import ember_calibration.analysis as analysis
from ember_calibration.archive_manifest import sha256_file


def synthetic_inputs():
    families = ["alpha"] * 250 + ["beta"] * 150 + ["gamma"] * 75 + ["delta"] * 25
    malicious_scores = np.concatenate(
        [
            np.linspace(0.2, 0.95, 250),
            np.linspace(0.3, 0.9, 150),
            np.linspace(0.4, 0.85, 75),
            np.linspace(0.45, 0.8, 25),
        ]
    )
    metadata = pd.DataFrame(
        {
            "family": families + [None] * 50,
            "label": [1] * len(families) + [0] * 50,
        }
    )
    predictions = np.concatenate([malicious_scores, np.linspace(0.01, 0.3, 50)])
    return metadata, predictions


def test_analysis_sensitivities_outputs_and_manifest(tmp_path, monkeypatch):
    metadata, predictions = synthetic_inputs()
    monkeypatch.setattr(
        analysis,
        "validate_reviewed_selection_inputs",
        lambda *args: SimpleNamespace(row_count=len(metadata)),
    )
    output_dir = tmp_path / "analysis"
    manifest = analysis.write_analysis_outputs(
        metadata,
        predictions,
        output_dir,
        threshold=0.5,
        primary_bins=15,
        primary_family_minimum=100,
        sensitivity_bins=(10, 15, 20, 30),
        sensitivity_family_minimums=(50, 100, 200),
        provenance={"implementation_commit": "0" * 40},
    )

    expected = {*analysis.EXPECTED_OUTPUTS, "analysis_manifest.json"}
    assert {path.name for path in output_dir.iterdir()} == expected
    aggregate = json.loads((output_dir / "aggregate_metrics.json").read_text())
    bin_sensitivity = pd.read_csv(output_dir / "calibration_bin_sensitivity.csv")
    assert bin_sensitivity["bin_count"].tolist() == [10, 15, 20, 30]
    primary = bin_sensitivity[bin_sensitivity["bin_count"] == 15].iloc[0]
    assert primary["aggregate_ece"] == pytest.approx(aggregate["ece"], abs=1e-15)
    assert primary["aggregate_mce"] == pytest.approx(aggregate["mce"], abs=1e-15)

    minimum_sensitivity = pd.read_csv(output_dir / "family_minimum_sensitivity.csv")
    assert minimum_sensitivity["minimum_malicious_family_count"].tolist() == [50, 100, 200]
    assert minimum_sensitivity["eligible_family_count"].tolist() == [3, 2, 1]
    assert minimum_sensitivity["eligible_malicious_row_count"].tolist() == [475, 400, 250]
    assert minimum_sensitivity["primary_top_10_false_negative_rate_overlap_count"].tolist() == [2, 2, 1]
    assert minimum_sensitivity["primary_top_10_ece_overlap_count"].tolist() == [2, 2, 1]

    families = pd.read_csv(output_dir / "family_metrics.csv")
    np.testing.assert_allclose(families["false_negative_rate"], 1.0 - families["accuracy"])
    for filename, record in manifest["outputs"].items():
        path = output_dir / filename
        assert path.stat().st_size == record["size_bytes"]
        assert sha256_file(path) == record["sha256"]
    for filename in ("reliability_diagram.png", "family_failures.png"):
        path = output_dir / filename
        assert path.stat().st_size > 0
        assert plt.imread(path).size > 0


@pytest.mark.parametrize(
    "arguments",
    [
        (0.5, 0, 100, (10, 15), (50, 100)),
        (0.5, 15, 0, (10, 15), (50, 100)),
        (0.5, 15, 100, (10, 10, 15), (50, 100)),
        (0.5, 15, 100, (10, 20), (50, 100)),
        (0.5, 15, 100, (10, 15), (50, 200)),
        (float("nan"), 15, 100, (10, 15), (50, 100)),
    ],
)
def test_invalid_analysis_parameters_fail_clearly(arguments):
    with pytest.raises(ValueError):
        analysis.validate_analysis_parameters(*arguments)
