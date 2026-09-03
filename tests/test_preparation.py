import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from ember_calibration.archive_manifest import sha256_file
from ember_calibration.selection import SelectionError
from ember_calibration.preparation import (
    METADATA_COLUMNS,
    validate_feature_file,
    validate_metadata_file,
    validate_vectorized_label_file,
)
from scripts.prepare_test_data import build_preparation_provenance, validate_aggregate_record_counts


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


def test_feature_file_requires_exact_shape_and_finite_float32_values(tmp_path):
    path = tmp_path / "features.dat"
    np.arange(6, dtype=np.float32).tofile(path)
    result = validate_feature_file(path, expected_rows=3, expected_features=2, chunk_rows=1)
    assert result == {
        "rows": 3,
        "feature_count": 2,
        "dtype": "float32",
        "size_bytes": 24,
        "finite_values": True,
    }


@pytest.mark.parametrize("values", [np.arange(5, dtype=np.float32), np.arange(7, dtype=np.float32)])
def test_feature_file_rejects_truncated_or_extra_bytes(tmp_path, values):
    path = tmp_path / "features.dat"
    values.tofile(path)
    with pytest.raises(ValueError, match="feature file size mismatch"):
        validate_feature_file(path, expected_rows=3, expected_features=2)


def test_feature_file_rejects_incorrect_feature_count(tmp_path):
    path = tmp_path / "features.dat"
    np.arange(6, dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="feature file size mismatch"):
        validate_feature_file(path, expected_rows=3, expected_features=3)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_feature_file_rejects_nonfinite_values(tmp_path, bad_value):
    path = tmp_path / "features.dat"
    values = np.arange(6, dtype=np.float32)
    values[2] = bad_value
    values.tofile(path)
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_feature_file(path, expected_rows=3, expected_features=2, chunk_rows=1)


def test_metadata_file_validates_rows_columns_order_and_repeat_profile(tmp_path):
    metadata = pd.DataFrame(
        [["a", 0, "Win32", None, 1], ["b", 1, "Win64", "family", 2]],
        columns=METADATA_COLUMNS,
    )
    path = tmp_path / "metadata.parquet"
    metadata.to_parquet(path, index=False)
    observed, result = validate_metadata_file(path, metadata, lambda frame: frame, expected_rows=2)
    assert observed.equals(metadata)
    assert result["selected_order_preserved"] is True


def test_metadata_file_rejects_row_or_column_mismatch(tmp_path):
    metadata = pd.DataFrame([["a", 0, "Win32", None, 1]], columns=METADATA_COLUMNS)
    path = tmp_path / "metadata.parquet"
    metadata.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="row count mismatch"):
        validate_metadata_file(path, metadata, lambda frame: frame, expected_rows=2)
    wrong_columns = metadata.drop(columns=["week_id"])
    with pytest.raises(ValueError, match="columns must be exactly"):
        validate_metadata_file(path, wrong_columns, lambda frame: frame, expected_rows=1)


def test_metadata_file_propagates_repeat_profile_failure(tmp_path):
    metadata = pd.DataFrame([["a", 0, "Win32", None, 1]], columns=METADATA_COLUMNS)
    path = tmp_path / "metadata.parquet"
    metadata.to_parquet(path, index=False)

    def reject_repeat_profile(frame):
        raise ValueError("repeat profile mismatch")

    with pytest.raises(ValueError, match="repeat profile mismatch"):
        validate_metadata_file(path, metadata, reject_repeat_profile, expected_rows=1)


def test_aggregate_counts_allow_multiple_files_per_type():
    observed = Counter({"Win32": 4, "Win64": 2, "Dot_Net": 1})
    validate_aggregate_record_counts(observed, {"Win32": 4, "Win64": 2, "Dot_Net": 1})


def test_aggregate_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="aggregate file-type"):
        validate_aggregate_record_counts(Counter({"Win32": 3}), {"Win32": 4})


def make_provenance_files(tmp_path):
    repository_root = tmp_path / "repository"
    selected = repository_root / "data/selected"
    processed = repository_root / "data/processed"
    models = repository_root / "models"
    selected.mkdir(parents=True)
    processed.mkdir(parents=True)
    models.mkdir(parents=True)
    selection_manifest = selected / "selection_manifest.json"
    selection_manifest.write_text("{}")
    selected_input = selected / "Win32_test/week.jsonl"
    selected_input.parent.mkdir()
    selected_input.write_text("{}\n")
    features = processed / "X_test.dat"
    np.array([1.0, 2.0], dtype=np.float32).tofile(features)
    labels = processed / "y_test.dat"
    np.array([1], dtype=np.int32).tofile(labels)
    metadata = processed / "test_metadata.parquet"
    metadata.write_bytes(b"synthetic metadata")
    preparation_manifest = processed / "preparation_manifest.json"
    preparation_manifest.write_text("{}")
    model = models / "EMBER2024_PE.model"
    model.write_bytes(b"synthetic model")
    return {
        "repository_root": repository_root,
        "selection_manifest": selection_manifest,
        "selected_input": selected_input,
        "features": features,
        "labels": labels,
        "metadata": metadata,
        "preparation_manifest": preparation_manifest,
        "model": model,
    }


def test_preparation_provenance_uses_explicit_relative_paths(tmp_path):
    paths = make_provenance_files(tmp_path)
    selection_document = {
        "selection_rule_name": "reviewed-rule",
        "repeated_hash_list_sha256": "abc",
        "extractor_version": "0.1.0",
        "extractor_dimension": 2,
    }
    input_observations = [
        {
            "file_type": "Win32",
            "path": paths["selected_input"],
            "rows": 1,
            "sha256": sha256_file(paths["selected_input"]),
        }
    ]
    preparation = build_preparation_provenance(
        paths["repository_root"],
        paths["selection_manifest"],
        selection_document,
        input_observations,
        1,
        2,
        paths["features"],
        paths["labels"],
        paths["metadata"],
        {"rows": 1, "feature_count": 2, "dtype": "float32", "finite_values": True},
        {"count": 1, "dtype": "int32", "allowed_values": True, "aligned_with_metadata": True},
        {
            "rows": 1,
            "columns": ["sha256", "label", "file_type", "family", "week_id"],
            "labels_aligned": True,
            "selected_order_preserved": True,
            "reviewed_repeat_profile": True,
        },
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
        1.0,
        "0" * 40,
        {
            "selection_manifest": "data/selected/selection_manifest.json",
            "output_dir": "data/processed",
            "execute_vectorization": True,
            "overwrite": False,
        },
        {"numpy": "test", "pandas": "test", "pyarrow": "test", "thrember": "0.1.0"},
    )
    assert preparation["selection_manifest"]["path"] == "data/selected/selection_manifest.json"
    assert preparation["inputs"][0]["path"] == "data/selected/Win32_test/week.jsonl"
    assert preparation["artifacts"]["features"]["path_base"] == (
        "preparation_manifest_directory"
    )
    combined_json = json.dumps({"preparation": preparation})
    assert str(tmp_path) not in combined_json
    assert not any(
        record["path"].startswith("/")
        for record in (
            preparation["selection_manifest"],
            preparation["inputs"][0],
        )
    )


def test_provenance_rejects_files_outside_repository(tmp_path):
    paths = make_provenance_files(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n")
    with pytest.raises(SelectionError, match="inside the repository"):
        build_preparation_provenance(
            paths["repository_root"],
            paths["selection_manifest"],
            {"selection_rule_name": "reviewed-rule", "repeated_hash_list_sha256": "abc"},
            [
                {
                    "file_type": "Win32",
                    "path": outside,
                    "rows": 1,
                    "sha256": sha256_file(outside),
                }
            ],
            1,
            2,
            paths["features"],
            paths["labels"],
            paths["metadata"],
            {"rows": 1, "feature_count": 2, "dtype": "float32", "finite_values": True},
            {"count": 1, "dtype": "int32", "allowed_values": True, "aligned_with_metadata": True},
            {
                "rows": 1,
                "columns": ["sha256", "label", "file_type", "family", "week_id"],
                "labels_aligned": True,
                "selected_order_preserved": True,
                "reviewed_repeat_profile": True,
            },
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
            1.0,
            "0" * 40,
            {
                "selection_manifest": "data/selected/selection_manifest.json",
                "output_dir": "data/processed",
                "execute_vectorization": True,
                "overwrite": False,
            },
            {"numpy": "test", "pandas": "test", "pyarrow": "test", "thrember": "0.1.0"},
        )
