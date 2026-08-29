import argparse
import hashlib
import json
from pathlib import Path

import pytest

from ember_calibration.artifact_integrity import ArtifactVerificationError
from scripts.download_data import parse_args, run_download


def expected_artifacts(contents):
    records = {}
    for filename, content in contents.items():
        is_model = filename == "EMBER2024_PE.model"
        records[filename] = {
            "artifact_type": "benchmark_model" if is_model else "dataset_archive",
            "repository": "models" if is_model else "dataset",
            "revision": "model-revision" if is_model else "dataset-revision",
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return records


def synthetic_inputs():
    return {
        "Win32_test.zip": b"win32 archive",
        "Win64_test.zip": b"win64 archive",
        "Dot_Net_test.zip": b"dotnet archive",
        "EMBER2024_PE.model": b"benchmark model",
    }


def arguments(tmp_path, download_only):
    return argparse.Namespace(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        execute=True,
        download_only=download_only,
    )


def downloader_for(contents):
    def download(**kwargs):
        path = Path(kwargs["local_dir"]) / str(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[path.name])
        return str(path)

    return download


def archive_mapping():
    return {
        "Win32_test.zip": "Win32",
        "Win64_test.zip": "Win64",
        "Dot_Net_test.zip": "Dot_Net",
    }


def test_download_only_requires_execute(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--model-dir",
                str(tmp_path / "models"),
                "--download-only",
            ]
        )


def test_download_only_never_calls_extraction_and_writes_integrity_manifest(tmp_path):
    contents = synthetic_inputs()

    def unexpected_extraction(*_args):
        raise AssertionError("download-only mode must not call extraction")

    manifest_path = run_download(
        arguments(tmp_path, download_only=True),
        downloader_for(contents),
        extract_function=unexpected_extraction,
        repository_root=tmp_path,
        expected_artifacts=expected_artifacts(contents),
        archive_mapping=archive_mapping(),
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == 1
    assert len(manifest["artifacts"]) == 4
    for artifact in manifest["artifacts"]:
        assert artifact["expected_size_bytes"] == artifact["observed_size_bytes"]
        assert artifact["expected_sha256"] == artifact["observed_sha256"]
        assert artifact["verification_status"] == "verified"
        assert artifact["extraction_occurred"] is False
        assert not Path(artifact["local_path"]).is_absolute()
        assert ".." not in Path(artifact["local_path"]).parts


def test_archive_verification_failure_prevents_all_extraction_and_manifest_writes(tmp_path):
    contents = synthetic_inputs()
    expected = expected_artifacts(contents)
    expected["Win64_test.zip"]["sha256"] = "0" * 64
    extraction_calls = []

    def record_extraction(*args):
        extraction_calls.append(args)
        return {}

    args = arguments(tmp_path, download_only=False)
    with pytest.raises(ArtifactVerificationError, match="Win64_test.zip: SHA-256 mismatch"):
        run_download(
            args,
            downloader_for(contents),
            extract_function=record_extraction,
            repository_root=tmp_path,
            expected_artifacts=expected,
            archive_mapping=archive_mapping(),
        )

    assert extraction_calls == []
    assert not (args.data_dir / "external_artifact_manifest.json").exists()
    assert not (args.data_dir / "download_manifest.json").exists()


def test_full_mode_extracts_only_after_all_artifacts_verify(tmp_path):
    contents = synthetic_inputs()
    extraction_calls = []

    def record_extraction(archive_path, _output_dir, file_type, _manifest_base):
        extraction_calls.append(archive_path.name)
        return {
            "archive_filename": archive_path.name,
            "archive_sha256": hashlib.sha256(contents[archive_path.name]).hexdigest(),
            "archive_size_bytes": len(contents[archive_path.name]),
            "assigned_file_type": file_type,
            "members": [{"member_path": "synthetic.jsonl", "is_jsonl": True}],
        }

    args = arguments(tmp_path, download_only=False)
    manifest_path = run_download(
        args,
        downloader_for(contents),
        extract_function=record_extraction,
        repository_root=tmp_path,
        expected_artifacts=expected_artifacts(contents),
        archive_mapping=archive_mapping(),
    )
    manifest = json.loads(manifest_path.read_text())

    assert extraction_calls == list(archive_mapping())
    extraction_status = {
        artifact["filename"]: artifact["extraction_occurred"]
        for artifact in manifest["artifacts"]
    }
    assert extraction_status["Win32_test.zip"] is True
    assert extraction_status["Win64_test.zip"] is True
    assert extraction_status["Dot_Net_test.zip"] is True
    assert extraction_status["EMBER2024_PE.model"] is False
