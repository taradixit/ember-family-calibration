import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ember_calibration.preparation as preparation_module
import scripts.prepare_test_data as prepare_cli
from ember_calibration.archive_manifest import sha256_file
from ember_calibration.preparation import METADATA_COLUMNS
from ember_calibration.selection import REPEATED_HASH_LIST_SHA256, SELECTION_RULE


def accept_metadata(metadata):
    return metadata


def make_context(tmp_path):
    repository_root = tmp_path / "repository"
    selected_dir = repository_root / "data/selected"
    selected_dir.mkdir(parents=True)
    selected_input = selected_dir / "week.jsonl"
    selected_input.write_text("{}\n")
    selection_document = {
        "selection_rule_name": SELECTION_RULE,
        "repeated_hash_list_sha256": REPEATED_HASH_LIST_SHA256,
        "extractor_version": "0.1.0",
        "extractor_dimension": 2,
        "selected_member_order": [
            {
                "file_type": "Win32",
                "selected_output_path": "data/selected/week.jsonl",
                "selected_row_count": 3,
                "selected_output_sha256": sha256_file(selected_input),
            }
        ],
    }
    selection_manifest = selected_dir / "selection_manifest.json"
    selection_manifest.write_text(json.dumps(selection_document))
    metadata = pd.DataFrame(
        [
            ["a", 0, "Win32", None, 1],
            ["b", 1, "Win32", "family", 2],
            ["c", 1, "Win32", "family", 3],
        ],
        columns=METADATA_COLUMNS,
    )
    return {
        "repository_root": repository_root,
        "selection_manifest": selection_manifest,
        "selection_document": selection_document,
        "selected_input": selected_input,
        "metadata": metadata,
        "output_dir": repository_root / "data/processed",
    }


def good_vectorizer(metadata):
    def vectorize(feature_path, label_path, paths, extractor, rows, label_field):
        np.arange(rows * 2, dtype=np.float32).tofile(feature_path)
        metadata["label"].to_numpy(dtype=np.int32).tofile(label_path)

    return vectorize


def run_preparation(context, vectorizer=None, overwrite=False, metadata_validator=accept_metadata):
    selected_input = context["selected_input"]
    return prepare_cli.execute_preparation(
        context["repository_root"],
        context["selection_manifest"],
        context["selection_document"],
        [("Win32", selected_input)],
        context["metadata"],
        [
            {
                "file_type": "Win32",
                "path": selected_input,
                "rows": 3,
                "sha256": sha256_file(selected_input),
            }
        ],
        context["output_dir"],
        vectorizer or good_vectorizer(context["metadata"]),
        object(),
        metadata_validator,
        overwrite=overwrite,
        expected_rows=3,
        expected_features=2,
        versions={
            "numpy": "test",
            "pandas": "test",
            "pyarrow": "test",
            "thrember": "0.1.0",
        },
        commit_identifier="0" * 40,
    )


def operation_directories(output_dir):
    return list(output_dir.parent.glob(".processed.staging-*")) + list(
        output_dir.parent.glob(".processed.backup-*")
    )


def test_successful_staged_preparation_publishes_complete_directory(tmp_path):
    context = make_context(tmp_path)
    manifest_path = run_preparation(context)
    output_dir = context["output_dir"]
    assert manifest_path == output_dir / "preparation_manifest.json"
    assert {path.name for path in output_dir.iterdir()} == {
        "X_test.dat",
        "y_test.dat",
        "test_metadata.parquet",
        "preparation_manifest.json",
    }
    document = json.loads(manifest_path.read_text())
    assert document["completion_status"] == "complete"
    assert document["rows"] == 3
    assert document["feature_count"] == 2
    assert document["artifacts"]["features"]["finite_values"] is True
    assert document["artifacts"]["labels"]["aligned_with_metadata"] is True
    assert document["artifacts"]["metadata"]["selected_order_preserved"] is True
    artifact_times = [
        (output_dir / name).stat().st_mtime_ns
        for name in ("X_test.dat", "y_test.dat", "test_metadata.parquet")
    ]
    assert manifest_path.stat().st_mtime_ns >= max(artifact_times)
    assert not operation_directories(output_dir)
    encoded = json.dumps(document)
    assert str(tmp_path) not in encoded
    assert ".processed.staging-" not in encoded


def test_vectorizer_failure_cleans_staging_and_leaves_no_final_output(tmp_path):
    context = make_context(tmp_path)

    def fail_vectorizer(*args, **kwargs):
        raise RuntimeError("vectorizer failed")

    with pytest.raises(RuntimeError, match="vectorizer failed"):
        run_preparation(context, vectorizer=fail_vectorizer)
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


def test_validation_failure_cleans_staging_and_leaves_no_final_output(tmp_path):
    context = make_context(tmp_path)

    def nan_vectorizer(feature_path, label_path, paths, extractor, rows, label_field):
        values = np.arange(rows * 2, dtype=np.float32)
        values[0] = np.nan
        values.tofile(feature_path)
        context["metadata"]["label"].to_numpy(dtype=np.int32).tofile(label_path)

    with pytest.raises(ValueError, match="NaN or infinity"):
        run_preparation(context, vectorizer=nan_vectorizer)
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


def test_label_order_mismatch_cleans_staging(tmp_path):
    context = make_context(tmp_path)

    def wrong_labels(feature_path, label_path, paths, extractor, rows, label_field):
        np.arange(rows * 2, dtype=np.float32).tofile(feature_path)
        np.array([1, 0, 1], dtype=np.int32).tofile(label_path)

    with pytest.raises(ValueError, match="not aligned"):
        run_preparation(context, vectorizer=wrong_labels)
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


def test_publication_failure_leaves_no_final_output(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    original_rename = preparation_module.rename_preparation_directory

    def fail_publish(source, destination):
        if source.name.startswith(".processed.staging-"):
            raise OSError("publication failed")
        original_rename(source, destination)

    monkeypatch.setattr(preparation_module, "rename_preparation_directory", fail_publish)
    with pytest.raises(OSError, match="publication failed"):
        run_preparation(context)
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


def test_existing_preparation_requires_overwrite(tmp_path):
    context = make_context(tmp_path)
    run_preparation(context)
    with pytest.raises(ValueError, match="pass --overwrite"):
        run_preparation(context)


def test_overwrite_publication_failure_restores_previous_complete_output(
    tmp_path, monkeypatch
):
    context = make_context(tmp_path)
    run_preparation(context)
    output_dir = context["output_dir"]
    previous = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    original_rename = preparation_module.rename_preparation_directory

    def fail_replacement(source, destination):
        if source.name.startswith(".processed.staging-") and destination == output_dir:
            raise OSError("replacement publication failed")
        original_rename(source, destination)

    monkeypatch.setattr(preparation_module, "rename_preparation_directory", fail_replacement)
    with pytest.raises(OSError, match="replacement publication failed"):
        run_preparation(context, overwrite=True)
    restored = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert restored == previous
    assert not operation_directories(output_dir)


def test_overwrite_requires_execute_vectorization():
    with pytest.raises(SystemExit):
        prepare_cli.parse_args(
            [
                "--selection-manifest",
                "data/selected/selection_manifest.json",
                "--output-dir",
                "data/processed",
                "--overwrite",
            ]
        )


def test_dry_run_validation_creates_no_processed_output(monkeypatch):
    repository_root = Path(prepare_cli.__file__).resolve().parents[1]
    output_dir = repository_root / "data/processed-dry-test"
    assert not output_dir.exists()
    selected_path = repository_root / "data/selected/selection_manifest.json"
    row = {"sha256": "a", "label": 0, "file_type": "Win32", "family": None, "week_id": 1}
    monkeypatch.setattr(
        prepare_cli,
        "load_completed_selection_manifest",
        lambda *args, **kwargs: ({}, [("Win32", selected_path)]),
    )
    monkeypatch.setattr(prepare_cli, "inspect_jsonl", lambda path: (1, [row]))
    monkeypatch.setattr(prepare_cli, "validate_aggregate_record_counts", lambda counts: None)
    monkeypatch.setattr(
        prepare_cli,
        "validate_reviewed_selection_metadata",
        lambda frame: SimpleNamespace(as_text=lambda: "synthetic validation passed"),
    )
    prepare_cli.main(
        [
            "--selection-manifest",
            "data/selected/selection_manifest.json",
            "--output-dir",
            "data/processed-dry-test",
        ]
    )
    assert not output_dir.exists()
