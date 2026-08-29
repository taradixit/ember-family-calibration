import hashlib

import pytest

import ember_calibration.artifact_integrity as artifact_integrity
from ember_calibration.artifact_integrity import (
    ArtifactVerificationError,
    verify_external_artifact,
)


def test_correct_size_and_hash_pass(tmp_path):
    path = tmp_path / "artifact.bin"
    content = b"verified content"
    path.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    result = verify_external_artifact(path, path.name, len(content), expected_hash)

    assert result["expected_size_bytes"] == len(content)
    assert result["observed_size_bytes"] == len(content)
    assert result["expected_sha256"] == expected_hash
    assert result["observed_sha256"] == expected_hash
    assert result["verification_status"] == "verified"


def test_incorrect_size_fails_before_hashing(tmp_path, monkeypatch):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"short")

    def unexpected_hash(_path):
        raise AssertionError("hashing must not run after a size mismatch")

    monkeypatch.setattr(artifact_integrity, "compute_sha256", unexpected_hash)
    with pytest.raises(ArtifactVerificationError, match="artifact.bin: size mismatch"):
        verify_external_artifact(path, path.name, 100, "0" * 64)


def test_incorrect_hash_fails(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"same size")
    with pytest.raises(ArtifactVerificationError, match="artifact.bin: SHA-256 mismatch"):
        verify_external_artifact(path, path.name, path.stat().st_size, "0" * 64)


def test_missing_file_fails(tmp_path):
    path = tmp_path / "missing.bin"
    with pytest.raises(ArtifactVerificationError, match="missing.bin: missing file"):
        verify_external_artifact(path, path.name, 0, hashlib.sha256(b"").hexdigest())


def test_directory_path_fails(tmp_path):
    with pytest.raises(ArtifactVerificationError, match="artifact.bin: not a regular file"):
        verify_external_artifact(tmp_path, "artifact.bin", 0, hashlib.sha256(b"").hexdigest())
