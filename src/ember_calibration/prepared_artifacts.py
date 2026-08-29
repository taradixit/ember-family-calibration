"""Validation for artifacts recorded by the preparation stage."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np

from .archive_manifest import sha256_file


def _resolve_recorded_path(manifest_path: Path, value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(value):
        raise ValueError(f"unsafe artifact path in preparation manifest: {value}")
    resolved = manifest_path.parent.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError(f"artifact path escapes preparation directory: {value}")
    return resolved


def load_prepared_artifacts(manifest_path: Path) -> dict[str, object]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = document.get("rows")
    feature_count = document.get("feature_count")
    if not isinstance(rows, int) or rows < 1:
        raise ValueError("preparation manifest has an invalid row count")
    if not isinstance(feature_count, int) or feature_count < 1:
        raise ValueError("preparation manifest has an invalid feature count")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preparation manifest has no artifact records")
    resolved: dict[str, Path] = {}
    for name in ("metadata", "features"):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"missing {name} artifact record")
        path = _resolve_recorded_path(manifest_path, record.get("path"))
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"{name} file size does not match preparation manifest")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{name} checksum does not match preparation manifest")
        resolved[name] = path
    expected_feature_bytes = rows * feature_count * np.dtype(np.float32).itemsize
    if resolved["features"].stat().st_size != expected_feature_bytes:
        raise ValueError(
            f"feature file byte-size mismatch: expected {expected_feature_bytes}, "
            f"found {resolved['features'].stat().st_size}"
        )
    return {
        "rows": rows,
        "feature_count": feature_count,
        "metadata": resolved["metadata"],
        "features": resolved["features"],
    }

