import json

import numpy as np
import pandas as pd
import pytest

from ember_calibration.archive_manifest import sha256_file
from ember_calibration.prepared_artifacts import load_prepared_artifacts
from ember_calibration.selection import REPEATED_HASH_LIST_SHA256, SELECTION_RULE


def accept_metadata(metadata):
    return metadata


def make_manifest(tmp_path):
    repository_root = tmp_path / "repository"
    selected_dir = repository_root / "data/selected"
    processed_dir = repository_root / "data/processed"
    selected_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    selected_input = selected_dir / "week.jsonl"
    selected_input.write_text("{}\n")
    selection_manifest = selected_dir / "selection_manifest.json"
    selection_manifest.write_text(
        json.dumps(
            {
                "selection_rule_name": SELECTION_RULE,
                "repeated_hash_list_sha256": REPEATED_HASH_LIST_SHA256,
                "selected_member_order": [
                    {
                        "file_type": "Win32",
                        "selected_output_path": "data/selected/week.jsonl",
                        "selected_row_count": 3,
                        "selected_output_sha256": sha256_file(selected_input),
                    }
                ],
            }
        )
    )
    metadata = pd.DataFrame(
        [
            ["a", 0, "Win32", None, 1],
            ["b", 1, "Win64", "family", 2],
            ["c", 1, "Dot_Net", "family", 3],
        ],
        columns=["sha256", "label", "file_type", "family", "week_id"],
    )
    metadata_path = processed_dir / "test_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    feature_path = processed_dir / "X_test.dat"
    np.arange(6, dtype=np.float32).tofile(feature_path)
    label_path = processed_dir / "y_test.dat"
    metadata["label"].to_numpy(dtype=np.int32).tofile(label_path)
    manifest = processed_dir / "preparation_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "completion_status": "complete",
                "start_time_utc": "2026-01-01T00:00:00Z",
                "finish_time_utc": "2026-01-01T00:00:01Z",
                "elapsed_seconds": 1.0,
                "code_commit": "0" * 40,
                "command_arguments": {
                    "selection_manifest": "data/selected/selection_manifest.json",
                    "output_dir": "data/processed",
                    "execute_vectorization": True,
                    "overwrite": False,
                },
                "path_bases": {
                    "repository_root": "repository root",
                    "preparation_manifest_directory": "directory containing this manifest",
                },
                "selection_manifest": {
                    "path": "data/selected/selection_manifest.json",
                    "path_base": "repository_root",
                    "sha256": sha256_file(selection_manifest),
                    "selection_rule_name": SELECTION_RULE,
                    "repeated_hash_list_sha256": REPEATED_HASH_LIST_SHA256,
                },
                "inputs": [
                    {
                        "file_type": "Win32",
                        "path": "data/selected/week.jsonl",
                        "path_base": "repository_root",
                        "rows": 3,
                        "sha256": sha256_file(selected_input),
                    }
                ],
                "rows": 3,
                "feature_count": 2,
                "extractor": {"version": "0.1.0", "dimension": 2},
                "dependency_versions": {
                    "numpy": "test",
                    "pandas": "test",
                    "pyarrow": "test",
                    "thrember": "0.1.0",
                },
                "feature_dtype": "float32",
                "label_dtype": "int32",
                "artifacts": {
                    "features": {
                        "path": feature_path.name,
                        "path_base": "preparation_manifest_directory",
                        "sha256": sha256_file(feature_path),
                        "size_bytes": feature_path.stat().st_size,
                        "rows": 3,
                        "feature_count": 2,
                        "dtype": "float32",
                        "finite_values": True,
                    },
                    "labels": {
                        "path": label_path.name,
                        "path_base": "preparation_manifest_directory",
                        "sha256": sha256_file(label_path),
                        "size_bytes": label_path.stat().st_size,
                        "count": 3,
                        "dtype": "int32",
                        "allowed_values": True,
                        "aligned_with_metadata": True,
                    },
                    "metadata": {
                        "path": metadata_path.name,
                        "path_base": "preparation_manifest_directory",
                        "sha256": sha256_file(metadata_path),
                        "size_bytes": metadata_path.stat().st_size,
                        "rows": 3,
                        "columns": list(metadata.columns),
                        "labels_aligned": True,
                        "selected_order_preserved": True,
                        "reviewed_repeat_profile": True,
                    },
                },
            }
        )
    )
    return manifest, repository_root, metadata


def load_small(manifest, repository_root):
    return load_prepared_artifacts(
        manifest,
        repository_root,
        expected_rows=3,
        expected_features=2,
        metadata_validator=accept_metadata,
    )


def test_prepared_artifacts_validate_all_three_artifacts(tmp_path):
    manifest, repository_root, _ = make_manifest(tmp_path)
    result = load_small(manifest, repository_root)
    assert result["rows"] == 3
    assert result["feature_count"] == 2
    assert result["labels"].name == "y_test.dat"


def test_prepared_artifacts_reject_wrong_exact_feature_size(tmp_path):
    manifest, repository_root, _ = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    feature_path = manifest.parent / "X_test.dat"
    feature_path.write_bytes(feature_path.read_bytes()[:-4])
    document["artifacts"]["features"]["size_bytes"] = feature_path.stat().st_size
    document["artifacts"]["features"]["sha256"] = sha256_file(feature_path)
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="feature file byte-size"):
        load_small(manifest, repository_root)


def test_prepared_artifacts_reject_checksum_mismatch(tmp_path):
    manifest, repository_root, _ = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["artifacts"]["metadata"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="metadata checksum"):
        load_small(manifest, repository_root)


def test_prepared_artifacts_reject_ambiguous_path_base(tmp_path):
    manifest, repository_root, _ = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["artifacts"]["features"]["path_base"] = "repository_root"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unsupported path base"):
        load_small(manifest, repository_root)


def test_prepared_artifacts_validate_label_checksum_and_alignment(tmp_path):
    manifest, repository_root, _ = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    label_path = manifest.parent / "y_test.dat"
    np.array([1, 0, 1], dtype=np.int32).tofile(label_path)
    document["artifacts"]["labels"]["size_bytes"] = label_path.stat().st_size
    document["artifacts"]["labels"]["sha256"] = sha256_file(label_path)
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="not aligned"):
        load_small(manifest, repository_root)


def test_prepared_artifacts_reject_selection_manifest_checksum_mismatch(tmp_path):
    manifest, repository_root, _ = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["selection_manifest"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="selection-manifest checksum"):
        load_small(manifest, repository_root)


def test_prepared_artifacts_reject_input_order_or_metadata_mismatch(tmp_path):
    manifest, repository_root, _ = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["inputs"][0]["rows"] = 2
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="order or metadata"):
        load_small(manifest, repository_root)


@pytest.mark.parametrize(("field", "value"), [("schema_version", 1), ("completion_status", "partial")])
def test_prepared_artifacts_require_supported_complete_manifest(tmp_path, field, value):
    manifest, repository_root, _ = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document[field] = value
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_small(manifest, repository_root)
