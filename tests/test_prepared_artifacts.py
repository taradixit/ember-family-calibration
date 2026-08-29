import json

import numpy as np
import pytest

from ember_calibration.archive_manifest import sha256_file
from ember_calibration.prepared_artifacts import load_prepared_artifacts


def make_manifest(tmp_path):
    metadata = tmp_path / "metadata.parquet"
    metadata.write_bytes(b"synthetic metadata")
    features = tmp_path / "features.dat"
    np.arange(6, dtype=np.float32).tofile(features)
    manifest = tmp_path / "preparation_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": 3,
                "feature_count": 2,
                "artifacts": {
                    "metadata": {
                        "path": metadata.name,
                        "sha256": sha256_file(metadata),
                        "size_bytes": metadata.stat().st_size,
                    },
                    "features": {
                        "path": features.name,
                        "sha256": sha256_file(features),
                        "size_bytes": features.stat().st_size,
                    },
                },
            }
        )
    )
    return manifest


def test_prepared_artifacts_validate_checksums_counts_and_feature_bytes(tmp_path):
    result = load_prepared_artifacts(make_manifest(tmp_path))
    assert result["rows"] == 3
    assert result["feature_count"] == 2


def test_prepared_artifacts_reject_wrong_exact_feature_size(tmp_path):
    manifest = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["feature_count"] = 3
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="feature file byte-size"):
        load_prepared_artifacts(manifest)


def test_prepared_artifacts_reject_checksum_mismatch(tmp_path):
    manifest = make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["artifacts"]["metadata"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="metadata checksum"):
        load_prepared_artifacts(manifest)

