import json
from collections import Counter

import numpy as np
import pytest

from ember_calibration.archive_manifest import sha256_file
from ember_calibration.selection import SelectionError
from ember_calibration.preparation import validate_vectorized_label_file
from scripts.prepare_test_data import build_preparation_provenance, validate_aggregate_record_counts
from scripts.run_inference import build_prediction_provenance


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


def test_preparation_and_prediction_provenance_use_explicit_relative_paths(tmp_path):
    paths = make_provenance_files(tmp_path)
    selection_document = {
        "selection_rule_name": "reviewed-rule",
        "repeated_hash_list_sha256": "abc",
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
    )
    prepared = {
        "rows": 1,
        "feature_count": 2,
        "features": paths["features"],
        "metadata": paths["metadata"],
    }
    prediction = build_prediction_provenance(
        paths["repository_root"],
        paths["model"],
        sha256_file(paths["model"]),
        paths["preparation_manifest"],
        prepared,
        "test-version",
        2,
    )
    assert preparation["selection_manifest"]["path"] == "data/selected/selection_manifest.json"
    assert preparation["inputs"][0]["path"] == "data/selected/Win32_test/week.jsonl"
    assert preparation["artifacts"]["features"]["path_base"] == (
        "preparation_manifest_directory"
    )
    assert prediction["model"]["path"] == "models/EMBER2024_PE.model"
    assert prediction["features"]["path"] == "X_test.dat"
    combined_json = json.dumps({"preparation": preparation, "prediction": prediction})
    assert str(tmp_path) not in combined_json
    assert not any(
        record["path"].startswith("/")
        for record in (
            preparation["selection_manifest"],
            preparation["inputs"][0],
            prediction["model"],
            prediction["preparation_manifest"],
            prediction["features"],
            prediction["metadata"],
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
        )
    with pytest.raises(SelectionError, match="inside the repository"):
        build_prediction_provenance(
            paths["repository_root"],
            outside,
            sha256_file(outside),
            paths["preparation_manifest"],
            {
                "rows": 1,
                "feature_count": 2,
                "features": paths["features"],
                "metadata": paths["metadata"],
            },
            "test-version",
            2,
        )
