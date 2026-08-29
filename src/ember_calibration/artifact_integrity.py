"""Exact size and checksum checks for external files."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

HASH_CHUNK_SIZE = 1024 * 1024


class ArtifactVerificationError(ValueError):
    """Raised when an external file does not match pinned metadata."""


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_external_artifact(
    path: Path,
    filename: str,
    expected_size_bytes: int,
    expected_sha256: str,
) -> dict[str, object]:
    """Verify one regular file against its exact pinned size and SHA-256."""
    if not path.exists():
        raise ArtifactVerificationError(f"{filename}: missing file")
    file_status = path.stat()
    if not stat.S_ISREG(file_status.st_mode):
        raise ArtifactVerificationError(f"{filename}: not a regular file")
    observed_size = file_status.st_size
    if observed_size != expected_size_bytes:
        raise ArtifactVerificationError(
            f"{filename}: size mismatch (expected {expected_size_bytes}, observed {observed_size})"
        )
    observed_sha256 = compute_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ArtifactVerificationError(f"{filename}: SHA-256 mismatch")
    return {
        "filename": filename,
        "expected_size_bytes": expected_size_bytes,
        "observed_size_bytes": observed_size,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "verification_status": "verified",
    }
