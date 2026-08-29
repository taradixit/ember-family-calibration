from collections import Counter

import numpy as np
import pytest

from ember_calibration.preparation import validate_vectorized_label_file
from scripts.prepare_test_data import validate_aggregate_record_counts


def test_int32_vectorized_labels_are_accepted(tmp_path):
    path = tmp_path / "labels.dat"
    np.array([0, 1, 1], dtype=np.int32).tofile(path)
    result = validate_vectorized_label_file(path, [0, 1, 1])
    np.testing.assert_array_equal(result, [0, 1, 1])


def test_float32_vectorized_labels_are_rejected(tmp_path):
    path = tmp_path / "labels.dat"
    np.array([0, 1], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="int32 encoding"):
        validate_vectorized_label_file(path, [0, 1])


def test_incorrect_vectorized_label_values_are_rejected(tmp_path):
    path = tmp_path / "labels.dat"
    np.array([0, 2], dtype=np.int32).tofile(path)
    with pytest.raises(ValueError, match="other than 0 and 1"):
        validate_vectorized_label_file(path, [0, 1])


def test_vectorized_label_size_mismatch_is_rejected(tmp_path):
    path = tmp_path / "labels.dat"
    np.array([0, 1], dtype=np.int32).tofile(path)
    with pytest.raises(ValueError, match="size mismatch"):
        validate_vectorized_label_file(path, [0, 1, 1])


def test_vectorized_label_alignment_mismatch_is_rejected(tmp_path):
    path = tmp_path / "labels.dat"
    np.array([1, 0], dtype=np.int32).tofile(path)
    with pytest.raises(ValueError, match="not aligned"):
        validate_vectorized_label_file(path, [0, 1])


def test_aggregate_counts_allow_multiple_files_per_type():
    observed = Counter({"Win32": 4, "Win64": 2, "Dot_Net": 1})
    validate_aggregate_record_counts(observed, {"Win32": 4, "Win64": 2, "Dot_Net": 1})


def test_aggregate_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="aggregate file-type"):
        validate_aggregate_record_counts(Counter({"Win32": 3}), {"Win32": 4})

