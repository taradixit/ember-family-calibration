import pandas as pd

from ember_calibration.family_analysis import family_metrics


def test_family_false_negative_rate_equals_one_minus_accuracy():
    metadata = pd.DataFrame({"family": ["alpha"] * 4, "label": [1] * 4})
    result = family_metrics(metadata, [0.9, 0.8, 0.4, 0.2], minimum_count=1, n_bins=4)
    assert result.loc[0, "accuracy"] == 0.5
    assert result.loc[0, "false_negative_rate"] == 0.5


def test_family_decision_threshold_changes_fnr_but_not_calibration():
    metadata = pd.DataFrame({"family": ["alpha"] * 3, "label": [1] * 3})
    predictions = [0.55, 0.7, 0.9]
    standard = family_metrics(metadata, predictions, minimum_count=1, threshold=0.5, n_bins=4)
    stricter = family_metrics(metadata, predictions, minimum_count=1, threshold=0.8, n_bins=4)
    assert standard.loc[0, "false_negative_rate"] != stricter.loc[0, "false_negative_rate"]
    assert standard.loc[0, "ece"] == stricter.loc[0, "ece"]


def test_family_minimum_size_uses_malicious_records_only():
    metadata = pd.DataFrame(
        {"family": ["large", "large", "small", "mixed", "mixed"], "label": [1, 1, 1, 1, 0]}
    )
    result = family_metrics(metadata, [0.9, 0.8, 0.7, 0.6, 0.1], minimum_count=2)
    assert result["family"].tolist() == ["large"]


def test_family_results_are_deterministic_and_sorted():
    metadata = pd.DataFrame({"family": ["zeta", "alpha", "zeta", "alpha"], "label": [1, 1, 1, 1]})
    predictions = [0.9, 0.8, 0.7, 0.6]
    first = family_metrics(metadata, predictions, minimum_count=2)
    second = family_metrics(metadata, predictions, minimum_count=2)
    assert first.equals(second)
    assert first["family"].tolist() == ["alpha", "zeta"]


def test_unusable_family_names_are_excluded_case_insensitively():
    metadata = pd.DataFrame(
        {
            "family": [None, " ", "UNKNOWN", "None", "NaN", "usable"],
            "label": [1, 1, 1, 1, 1, 1],
        }
    )
    result = family_metrics(metadata, [0.9] * 6, minimum_count=1)
    assert result["family"].tolist() == ["usable"]


def test_family_sort_is_fnr_then_ece_then_name():
    metadata = pd.DataFrame({"family": ["low", "high", "low", "high"], "label": [1, 1, 1, 1]})
    result = family_metrics(metadata, [0.9, 0.4, 0.8, 0.3], minimum_count=2)
    assert result["family"].tolist() == ["high", "low"]
