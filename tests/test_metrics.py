import numpy as np
import pytest

from ember_calibration.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    predicted_label_confidence,
    reliability_bins,
    roc_auc,
)


def test_perfect_predictions_are_perfectly_calibrated():
    labels = [0, 0, 1, 1]
    predictions = [0.0, 0.0, 1.0, 1.0]
    assert accuracy(labels, predictions) == 1.0
    assert brier_score(labels, predictions) == 0.0
    assert expected_calibration_error(labels, predictions, n_bins=5) == 0.0
    assert maximum_calibration_error(labels, predictions, n_bins=5) == 0.0


def test_clearly_miscalibrated_predictions_have_large_error():
    labels = [0, 0, 1, 1]
    predictions = [1.0, 1.0, 0.0, 0.0]
    assert accuracy(labels, predictions) == 0.0
    assert expected_calibration_error(labels, predictions, n_bins=5) == 1.0
    assert maximum_calibration_error(labels, predictions, n_bins=5) == 1.0


def test_four_bins_span_half_to_one_with_correct_boundaries_and_counts():
    predictions = [0.5, 0.625, 0.75, 0.875, 1.0]
    bins = reliability_bins([1] * len(predictions), predictions, n_bins=4)
    assert [(item["lower"], item["upper"]) for item in bins] == [
        (0.5, 0.625),
        (0.625, 0.75),
        (0.75, 0.875),
        (0.875, 1.0),
    ]
    assert [item["count"] for item in bins] == [1, 1, 1, 2]


def test_every_internal_boundary_enters_the_bin_on_its_right():
    boundaries = np.linspace(0.5, 1.0, 5)
    bins = reliability_bins([1] * len(boundaries), boundaries, n_bins=4)
    assert [item["count"] for item in bins] == [1, 1, 1, 2]


def test_confidence_half_and_one_are_in_first_and_last_bins():
    bins = reliability_bins([1, 1], [0.5, 1.0], n_bins=4)
    assert bins[0]["count"] == 1
    assert bins[-1]["count"] == 1


def test_empty_bins_are_explicit_and_do_not_affect_metrics():
    bins = reliability_bins([1], [1.0], n_bins=4)
    assert bins[0]["count"] == 0
    assert bins[0]["accuracy"] is None
    assert expected_calibration_error([1], [1.0], n_bins=4) == 0.0


def test_manually_calculated_ece_and_mce():
    labels = [1, 0, 1, 0, 1]
    predictions = [0.5, 0.6, 0.7, 0.2, 0.1]
    assert expected_calibration_error(labels, predictions, n_bins=4) == pytest.approx(0.3)
    assert maximum_calibration_error(labels, predictions, n_bins=4) == pytest.approx(0.9)


def test_predicted_label_confidence_uses_standard_binary_prediction():
    np.testing.assert_allclose(predicted_label_confidence([0.2, 0.5, 0.9]), [0.8, 0.5, 0.9])


def test_auc_perfect_and_reversed_rankings():
    labels = [0, 0, 1, 1]
    assert roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_tied_predictions():
    assert roc_auc([0, 1, 0, 1], [0.2, 0.8, 0.8, 0.8]) == pytest.approx(0.75)


def test_auc_large_input_uses_scalable_library_implementation():
    labels = np.tile([0, 1], 50_000)
    predictions = np.tile([0.25, 0.75], 50_000)
    assert roc_auc(labels, predictions) == 1.0


def test_auc_requires_both_classes():
    with pytest.raises(ValueError, match="positive and one negative"):
        roc_auc([1, 1], [0.2, 0.8])


def test_length_mismatch_fails():
    with pytest.raises(ValueError, match="same length"):
        accuracy([0, 1], [0.2])


@pytest.mark.parametrize("predictions", [[-0.1], [1.1], [np.nan], [np.inf]])
def test_invalid_prediction_values_fail(predictions):
    with pytest.raises(ValueError):
        accuracy([0], predictions)


def test_metrics_are_deterministic():
    labels = [0, 1, 1, 0]
    predictions = [0.1, 0.8, 0.6, 0.3]
    first = expected_calibration_error(labels, predictions, n_bins=4)
    second = expected_calibration_error(labels, predictions, n_bins=4)
    assert first == second

