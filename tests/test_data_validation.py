import hashlib

import numpy as np
import pandas as pd
import pytest

from ember_calibration.data_validation import (
    ValidationError,
    build_report,
    filter_by_minimum_count,
    validate_metadata,
    validate_predictions,
    validate_reviewed_selection_metadata,
)


def metadata(rows):
    return pd.DataFrame(rows, columns=["sha256", "label", "file_type", "family"])


def test_report_contains_required_audit_counts():
    frame = metadata([["a", 0, "Win32", None], ["a", 0, "Win32", None], ["b", 1, ".NET", "x"]])
    report = build_report(frame)
    assert report.row_count == 3
    assert report.unique_hash_count == 2
    assert report.duplicate_multiplicities == {1: 1, 2: 1}
    assert report.file_type_counts == {"Win32": 2, "Dot_Net": 1}


def test_missing_hash_fails():
    frame = metadata([[None, 1, "Win32", "x"]])
    with pytest.raises(ValidationError, match="missing or blank"):
        validate_metadata(frame, expected_records=1, expected_file_types={"Win32": 1})


def test_duplicate_hash_fails_without_deduplicating():
    frame = metadata([["a", 1, "Win32", "x"], ["a", 1, "Win32", "x"]])
    with pytest.raises(ValidationError, match="duplicate SHA-256"):
        validate_metadata(frame, expected_records=2, expected_file_types={"Win32": 2})
    assert len(frame) == 2


def test_completely_doubled_dataset_fails_loudly():
    base = metadata([["a", 0, "Win32", None], ["b", 1, "Win64", "x"]])
    doubled = pd.concat([base, base], ignore_index=True)
    with pytest.raises(ValidationError, match="systematically doubled"):
        validate_metadata(doubled, expected_records=2, expected_file_types={"Win32": 1, "Win64": 1})


def test_file_type_count_mismatch_fails():
    frame = metadata([["a", 0, "Win32", None], ["b", 1, "Win32", "x"]])
    with pytest.raises(ValidationError, match="file-type counts"):
        validate_metadata(frame, expected_records=2, expected_file_types={"Win32": 1, "Win64": 1})


def test_prediction_length_mismatch_fails():
    with pytest.raises(ValidationError, match="length mismatch"):
        validate_predictions([0.1], expected_length=2)


def test_prediction_validation_rejects_nonfinite_and_out_of_range():
    for values in ([np.nan], [np.inf], [-0.01], [1.01]):
        with pytest.raises(ValidationError):
            validate_predictions(values)


def test_minimum_count_filtering_keeps_only_eligible_groups():
    frame = pd.DataFrame({"family": ["a", "a", "b"], "label": [1, 1, 1]})
    filtered = filter_by_minimum_count(frame, "family", 2)
    assert filtered["family"].tolist() == ["a", "a"]


def test_reviewed_selection_validation_allows_only_expected_cross_week_repeat():
    frame = pd.DataFrame(
        [
            ["repeat", 1, "Win32", "family-a", 1],
            ["repeat", 1, "Win32", "family-a", 2],
            ["unique", 0, "Win64", None, 3],
        ],
        columns=["sha256", "label", "file_type", "family", "week_id"],
    )
    repeated_digest = hashlib.sha256(b"repeat").hexdigest()
    report = validate_reviewed_selection_metadata(
        frame,
        expected_records=3,
        expected_file_types={"Win32": 2, "Win64": 1},
        expected_unique_hashes=2,
        expected_multiplicities={1: 1, 2: 1},
        expected_repeated_hash_digest=repeated_digest,
    )
    assert report.unique_hash_count == 2


def test_reviewed_selection_validation_rejects_repeat_conflict():
    frame = pd.DataFrame(
        [
            ["repeat", 1, "Win32", "family-a", 1],
            ["repeat", 0, "Win32", "family-a", 2],
        ],
        columns=["sha256", "label", "file_type", "family", "week_id"],
    )
    repeated_digest = hashlib.sha256(b"repeat").hexdigest()
    with pytest.raises(ValidationError, match="label conflicts"):
        validate_reviewed_selection_metadata(
            frame,
            expected_records=2,
            expected_file_types={"Win32": 2},
            expected_unique_hashes=1,
            expected_multiplicities={2: 1},
            expected_repeated_hash_digest=repeated_digest,
        )
