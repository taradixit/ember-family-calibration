import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ember_calibration.inference as inference_module
import scripts.run_inference as inference_cli
from ember_calibration.archive_manifest import sha256_file
from ember_calibration.inference import positive_batch_size, validate_prediction_array


class FakeModel:
    def __init__(self, feature_count=2, prediction_function=None):
        self.feature_count = feature_count
        self.prediction_function = prediction_function or (lambda batch: batch[:, 0])
        self.batch_lengths = []

    def num_feature(self):
        return self.feature_count

    def predict(self, batch):
        self.batch_lengths.append(len(batch))
        return self.prediction_function(batch)


def make_context(tmp_path, rows=5, feature_count=2):
    repository_root = tmp_path / "repository"
    processed_dir = repository_root / "data/processed"
    model_dir = repository_root / "models"
    processed_dir.mkdir(parents=True)
    model_dir.mkdir()
    features = np.zeros((rows, feature_count), dtype=np.float32)
    features[:, 0] = np.linspace(0.1, 0.9, rows)
    feature_path = processed_dir / "X_test.dat"
    features.tofile(feature_path)
    preparation_manifest = processed_dir / "preparation_manifest.json"
    preparation_manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "features": {"sha256": sha256_file(feature_path)},
                    "labels": {"sha256": "1" * 64},
                    "metadata": {"sha256": "2" * 64},
                }
            }
        )
    )
    model_path = model_dir / "EMBER2024_PE.model"
    model_path.write_bytes(b"synthetic model")
    return {
        "repository_root": repository_root,
        "features": features,
        "feature_path": feature_path,
        "preparation_manifest": preparation_manifest,
        "model_path": model_path,
        "model_checksum": sha256_file(model_path),
        "prepared": {"rows": rows, "feature_count": feature_count, "features": feature_path},
        "output_dir": repository_root / "results/inference",
    }


def run_inference(context, model=None, batch_size=2, overwrite=False, validator=None):
    return inference_cli.execute_inference(
        context["repository_root"],
        context["model_path"],
        context["model_checksum"],
        context["preparation_manifest"],
        context["prepared"],
        context["output_dir"],
        model or FakeModel(),
        "test-lightgbm",
        batch_size=batch_size,
        overwrite=overwrite,
        commit_identifier="0" * 40,
        manifest_validator=validator or (lambda path: path),
    )


def operation_directories(output_dir):
    return list(output_dir.parent.glob(".inference.staging-*")) + list(
        output_dir.parent.glob(".inference.backup-*")
    )


def test_successful_multi_batch_prediction_preserves_order_and_final_partial_batch(tmp_path):
    context = make_context(tmp_path)
    model = FakeModel()
    manifest_path = run_inference(context, model=model, batch_size=2)
    output_dir = context["output_dir"]
    predictions = np.load(output_dir / "predictions.npy", allow_pickle=False)
    np.testing.assert_allclose(predictions, context["features"][:, 0])
    assert predictions.dtype == np.float64
    assert model.batch_lengths == [2, 2, 1]
    assert {path.name for path in output_dir.iterdir()} == {
        "predictions.npy",
        "inference_manifest.json",
    }
    document = json.loads(manifest_path.read_text())
    assert document["prediction_batch_count"] == 3
    assert document["threshold_applied"] is False
    assert document["prediction"]["shape"] == [5]
    assert manifest_path.stat().st_mtime_ns >= (output_dir / "predictions.npy").stat().st_mtime_ns
    assert not operation_directories(output_dir)
    assert str(tmp_path) not in json.dumps(document)
    assert ".inference.staging-" not in json.dumps(document)


def test_batch_size_larger_than_row_count_uses_one_batch(tmp_path):
    context = make_context(tmp_path, rows=3)
    model = FakeModel()
    run_inference(context, model=model, batch_size=10)
    assert model.batch_lengths == [3]


@pytest.mark.parametrize("value", [0, -1, "0", "-4", 1.5, "1.5", True])
def test_batch_size_must_be_positive(value):
    with pytest.raises(ValueError, match="positive integer"):
        positive_batch_size(value)


def test_cli_batch_size_defaults_to_ten_thousand():
    args = inference_cli.parse_args(
        [
            "--model",
            "models/EMBER2024_PE.model",
            "--preparation-manifest",
            "data/processed/preparation_manifest.json",
            "--output-dir",
            "results/inference",
        ]
    )
    assert args.batch_size == 10_000


def test_model_feature_count_mismatch_is_rejected_before_staging(tmp_path):
    context = make_context(tmp_path)
    with pytest.raises(ValueError, match="feature-count mismatch"):
        run_inference(context, model=FakeModel(feature_count=3))
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


@pytest.mark.parametrize(
    ("prediction_function", "message"),
    [
        (lambda batch: batch[:-1, 0], "prediction count mismatch"),
        (lambda batch: batch[:, :1], "one-dimensional"),
        (lambda batch: np.full(len(batch), np.nan), "non-finite"),
        (lambda batch: np.full(len(batch), np.inf), "non-finite"),
        (lambda batch: np.full(len(batch), -np.inf), "non-finite"),
        (lambda batch: np.full(len(batch), -0.1), r"in \[0, 1\]"),
        (lambda batch: np.full(len(batch), 1.1), r"in \[0, 1\]"),
    ],
)
def test_invalid_batch_predictions_clean_staging(tmp_path, prediction_function, message):
    context = make_context(tmp_path)
    model = FakeModel(prediction_function=prediction_function)
    with pytest.raises(ValueError, match=message):
        run_inference(context, model=model)
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


def test_model_failure_cleans_staging(tmp_path):
    context = make_context(tmp_path)

    def fail_model(batch):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        run_inference(context, model=FakeModel(prediction_function=fail_model))
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


def test_missing_feature_file_cleans_staging(tmp_path):
    context = make_context(tmp_path)
    context["feature_path"].unlink()
    with pytest.raises(FileNotFoundError):
        run_inference(context)
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


def test_prediction_validator_requires_float64_shape_and_range():
    validate_prediction_array(np.array([0.1, 0.9], dtype=np.float64), 2, require_float64=True)
    with pytest.raises(ValueError, match="float64"):
        validate_prediction_array(np.array([0.1], dtype=np.float32), 1, require_float64=True)
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_prediction_array(np.array([[0.1]], dtype=np.float64), 1)


def test_publication_failure_leaves_no_final_output(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    original_rename = inference_module.rename_inference_directory

    def fail_publish(source, destination):
        if source.name.startswith(".inference.staging-"):
            raise OSError("publication failed")
        original_rename(source, destination)

    monkeypatch.setattr(inference_module, "rename_inference_directory", fail_publish)
    with pytest.raises(OSError, match="publication failed"):
        run_inference(context)
    assert not context["output_dir"].exists()
    assert not operation_directories(context["output_dir"])


def test_existing_output_requires_overwrite(tmp_path):
    context = make_context(tmp_path)
    run_inference(context)
    with pytest.raises(ValueError, match="pass --overwrite"):
        run_inference(context)


def test_overwrite_failure_restores_previous_complete_output(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    run_inference(context)
    output_dir = context["output_dir"]
    previous = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    original_rename = inference_module.rename_inference_directory

    def fail_replacement(source, destination):
        if source.name.startswith(".inference.staging-") and destination == output_dir:
            raise OSError("replacement failed")
        original_rename(source, destination)

    monkeypatch.setattr(inference_module, "rename_inference_directory", fail_replacement)
    with pytest.raises(OSError, match="replacement failed"):
        run_inference(context, overwrite=True)
    restored = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    assert restored == previous
    assert not operation_directories(output_dir)


def test_overwrite_requires_execute():
    with pytest.raises(SystemExit):
        inference_cli.parse_args(
            [
                "--model",
                "models/EMBER2024_PE.model",
                "--preparation-manifest",
                "data/processed/preparation_manifest.json",
                "--output-dir",
                "results/inference",
                "--overwrite",
            ]
        )


def test_dry_run_preflight_creates_no_prediction_output(tmp_path, monkeypatch):
    repository_root = Path(inference_cli.__file__).resolve().parents[1]
    output_dir = repository_root / "results/inference-dry-test"
    assert not output_dir.exists()
    prepared = {
        "rows": 1,
        "feature_count": 2,
        "features": tmp_path / "features.dat",
        "metadata": tmp_path / "metadata.parquet",
    }
    np.array([0.1, 0.2], dtype=np.float32).tofile(prepared["features"])
    pd.DataFrame(
        [["a", 0, "Win32", None, 1]],
        columns=["sha256", "label", "file_type", "family", "week_id"],
    ).to_parquet(prepared["metadata"], index=False)
    monkeypatch.setattr(inference_cli, "load_prepared_artifacts", lambda *args: prepared)
    monkeypatch.setattr(inference_cli, "sha256_file", lambda path: "0" * 64)
    monkeypatch.setattr(inference_cli, "HISTORICAL_MODEL_SHA256", "0" * 64)
    monkeypatch.setattr(
        inference_cli,
        "validate_reviewed_selection_metadata",
        lambda metadata: type("Report", (), {"as_text": lambda self: "valid"})(),
    )
    inference_cli.main(
        [
            "--model",
            "models/EMBER2024_PE.model",
            "--preparation-manifest",
            "data/processed/preparation_manifest.json",
            "--output-dir",
            "results/inference-dry-test",
        ]
    )
    assert not output_dir.exists()
