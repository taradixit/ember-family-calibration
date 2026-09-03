import json

import numpy as np
import pandas as pd
import pytest

import scripts.analyze_results as analysis_cli
from ember_calibration.archive_manifest import sha256_file
from ember_calibration.inference_artifacts import (
    PREDICTION_MEANING,
    load_inference_artifacts,
)
from ember_calibration.upstream import MODEL_REVISION


def make_manifest(tmp_path):
    repository_root = tmp_path / "repository"
    model_dir = repository_root / "models"
    processed_dir = repository_root / "data/processed"
    inference_dir = repository_root / "results/inference"
    model_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    inference_dir.mkdir(parents=True)
    model_path = model_dir / "EMBER2024_PE.model"
    model_path.write_bytes(b"synthetic model")
    metadata_path = processed_dir / "test_metadata.parquet"
    metadata_path.write_bytes(b"synthetic metadata")
    preparation_manifest = processed_dir / "preparation_manifest.json"
    preparation_manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "features": {"sha256": "1" * 64},
                    "labels": {"sha256": "2" * 64},
                    "metadata": {"sha256": "3" * 64},
                }
            }
        )
    )
    prediction_path = inference_dir / "predictions.npy"
    np.save(prediction_path, np.array([0.1, 0.5, 0.9], dtype=np.float64))
    model_checksum = sha256_file(model_path)
    manifest_path = inference_dir / "inference_manifest.json"
    document = {
        "schema_version": 1,
        "completion_status": "complete",
        "start_time_utc": "2026-01-01T00:00:00Z",
        "finish_time_utc": "2026-01-01T00:00:01Z",
        "elapsed_seconds": 1.0,
        "code_commit": "0" * 40,
        "command_arguments": {
            "model": "models/EMBER2024_PE.model",
            "preparation_manifest": "data/processed/preparation_manifest.json",
            "output_dir": "results/inference",
            "batch_size": 2,
            "execute": True,
            "overwrite": False,
        },
        "path_bases": {
            "repository_root": "repository root",
            "inference_manifest_directory": "directory containing this manifest",
        },
        "batch_size": 2,
        "prediction_batch_count": 2,
        "rows": 3,
        "feature_count": 2,
        "prediction_meaning": PREDICTION_MEANING,
        "threshold_applied": False,
        "model": {
            "repository_revision": MODEL_REVISION,
            "path": "models/EMBER2024_PE.model",
            "path_base": "repository_root",
            "sha256": model_checksum,
        },
        "lightgbm_version": "test",
        "model_feature_count": 2,
        "preparation_manifest": {
            "path": "data/processed/preparation_manifest.json",
            "path_base": "repository_root",
            "sha256": sha256_file(preparation_manifest),
        },
        "preparation_artifact_sha256": {
            "features": "1" * 64,
            "labels": "2" * 64,
            "metadata": "3" * 64,
        },
        "prediction": {
            "filename": "predictions.npy",
            "path": "predictions.npy",
            "path_base": "inference_manifest_directory",
            "dtype": "float64",
            "shape": [3],
            "size_bytes": prediction_path.stat().st_size,
            "sha256": sha256_file(prediction_path),
            "minimum": 0.1,
            "maximum": 0.9,
            "finite_values": True,
            "range_valid": True,
            "exact_row_count": True,
        },
    }
    manifest_path.write_text(json.dumps(document))
    return {
        "repository_root": repository_root,
        "manifest": manifest_path,
        "document": document,
        "model": model_path,
        "model_checksum": model_checksum,
        "preparation_manifest": preparation_manifest,
        "metadata": metadata_path,
        "predictions": prediction_path,
    }


def load_small(context, preparation_calls=None, expected_publication_directory=None):
    def preparation_loader(*args, **kwargs):
        if preparation_calls is not None:
            preparation_calls.append((args, kwargs))
        return {"metadata": context["metadata"]}

    arguments = {
        "expected_rows": 3,
        "expected_features": 2,
        "expected_model_sha256": context["model_checksum"],
        "preparation_loader": preparation_loader,
    }
    if expected_publication_directory is not None:
        arguments["expected_publication_directory"] = expected_publication_directory
    return load_inference_artifacts(
        context["manifest"],
        context["repository_root"],
        **arguments,
    )


def rewrite_manifest(context):
    context["manifest"].write_text(json.dumps(context["document"]))


def refresh_prediction_record(context):
    predictions = np.load(context["predictions"], allow_pickle=False)
    record = context["document"]["prediction"]
    record["size_bytes"] = context["predictions"].stat().st_size
    record["sha256"] = sha256_file(context["predictions"])
    record["minimum"] = float(predictions.min())
    record["maximum"] = float(predictions.max())
    rewrite_manifest(context)


def move_to_staging(context):
    staging_dir = context["repository_root"] / "results/.inference.staging-test"
    context["manifest"].parent.replace(staging_dir)
    context["manifest"] = staging_dir / "inference_manifest.json"
    context["predictions"] = staging_dir / "predictions.npy"
    return staging_dir


def test_loader_validates_manifest_chain_and_returns_paths(tmp_path):
    context = make_manifest(tmp_path)
    preparation_calls = []
    result = load_small(context, preparation_calls)
    assert result["predictions"] == context["predictions"]
    assert result["metadata"] == context["metadata"]
    assert len(preparation_calls) == 1


def test_loader_validates_staged_manifest_for_explicit_publication_directory(tmp_path):
    context = make_manifest(tmp_path)
    staging_dir = move_to_staging(context)
    publication_directory = context["repository_root"] / "results/inference"
    result = load_small(
        context,
        expected_publication_directory=publication_directory,
    )
    assert result["predictions"] == staging_dir / "predictions.npy"


def test_loader_rejects_staged_manifest_without_publication_context(tmp_path):
    context = make_manifest(tmp_path)
    move_to_staging(context)
    with pytest.raises(ValueError, match="command arguments"):
        load_small(context)


def test_loader_rejects_wrong_staged_publication_directory(tmp_path):
    context = make_manifest(tmp_path)
    move_to_staging(context)
    with pytest.raises(ValueError, match="command arguments"):
        load_small(
            context,
            expected_publication_directory=context["repository_root"] / "results/other",
        )


def test_loader_rejects_publication_directory_outside_repository(tmp_path):
    context = make_manifest(tmp_path)
    move_to_staging(context)
    with pytest.raises(ValueError, match="inside the repository"):
        load_small(
            context,
            expected_publication_directory=tmp_path / "outside",
        )


def test_loader_rejects_prediction_checksum_mismatch(tmp_path):
    context = make_manifest(tmp_path)
    context["document"]["prediction"]["sha256"] = "0" * 64
    rewrite_manifest(context)
    with pytest.raises(ValueError, match="prediction checksum"):
        load_small(context)


def test_loader_rejects_preparation_manifest_checksum_mismatch(tmp_path):
    context = make_manifest(tmp_path)
    context["document"]["preparation_manifest"]["sha256"] = "0" * 64
    rewrite_manifest(context)
    with pytest.raises(ValueError, match="preparation-manifest checksum"):
        load_small(context)


def test_loader_rejects_model_checksum_mismatch(tmp_path):
    context = make_manifest(tmp_path)
    context["model"].write_bytes(b"changed model")
    with pytest.raises(ValueError, match="model checksum"):
        load_small(context)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (("prediction", "path"), "unexpected prediction path"),
        (("model", "path"), "repository-relative"),
    ],
)
def test_loader_rejects_absolute_paths(tmp_path, record, message):
    context = make_manifest(tmp_path)
    context["document"][record[0]][record[1]] = "/private/untrusted"
    rewrite_manifest(context)
    with pytest.raises(ValueError, match=message):
        load_small(context)


def test_loader_rejects_output_directory_that_does_not_contain_manifest(tmp_path):
    context = make_manifest(tmp_path)
    context["document"]["command_arguments"]["output_dir"] = "results/other"
    rewrite_manifest(context)
    with pytest.raises(ValueError, match="command arguments"):
        load_small(context)


def test_loader_rejects_wrong_prediction_dtype(tmp_path):
    context = make_manifest(tmp_path)
    np.save(context["predictions"], np.array([0.1, 0.5, 0.9], dtype=np.float32))
    refresh_prediction_record(context)
    with pytest.raises(ValueError, match="float64"):
        load_small(context)


def test_loader_rejects_wrong_prediction_shape(tmp_path):
    context = make_manifest(tmp_path)
    np.save(context["predictions"], np.array([[0.1], [0.5], [0.9]], dtype=np.float64))
    refresh_prediction_record(context)
    with pytest.raises(ValueError, match="one-dimensional"):
        load_small(context)


def test_analysis_cli_loads_both_inputs_through_inference_manifest(tmp_path, monkeypatch):
    repository_root = analysis_cli.Path(analysis_cli.__file__).resolve().parents[1]
    metadata_path = tmp_path / "metadata.parquet"
    prediction_path = tmp_path / "predictions.npy"
    pd.DataFrame({"label": [0]}).to_parquet(metadata_path, index=False)
    np.save(prediction_path, np.array([0.1], dtype=np.float64))
    preparation_path = tmp_path / "preparation_manifest.json"
    model_path = tmp_path / "model.txt"
    preparation_path.write_text("{}")
    model_path.write_text("model")
    observed = {}

    def load_manifest(path, root):
        observed["manifest"] = path
        observed["root"] = root
        return {
            "metadata": metadata_path,
            "predictions": prediction_path,
            "preparation_manifest": preparation_path,
            "model": model_path,
        }

    def capture_outputs(
        metadata,
        predictions,
        output_dir,
        threshold,
        bins,
        minimum_count,
        sensitivity_bins,
        sensitivity_minimums,
        provenance,
    ):
        observed["metadata"] = metadata
        observed["predictions"] = np.asarray(predictions).copy()
        observed["output_dir"] = output_dir
        observed["sensitivity_bins"] = sensitivity_bins
        observed["sensitivity_minimums"] = sensitivity_minimums

    monkeypatch.setattr(analysis_cli, "load_inference_artifacts", load_manifest)
    monkeypatch.setattr(analysis_cli, "write_analysis_outputs", capture_outputs)
    monkeypatch.setattr(
        analysis_cli,
        "repository_relative_path",
        lambda path, root: analysis_cli.Path(path).name,
    )
    manifest = repository_root / "results/inference/inference_manifest.json"
    output_dir = repository_root / "results/analysis"
    analysis_cli.main(
        [
            "--inference-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert observed["manifest"] == manifest
    assert observed["metadata"]["label"].tolist() == [0]
    np.testing.assert_array_equal(observed["predictions"], [0.1])
    assert observed["output_dir"] == output_dir
    assert observed["sensitivity_bins"] == (10, 15, 20, 30)
    assert observed["sensitivity_minimums"] == (50, 100, 200)
