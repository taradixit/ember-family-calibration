"""Verified upstream identifiers used by future workflow stages."""

THREMBER_GIT_REVISION = "0ef753e81d98bf209f71b03cd331dfc190b5b54d"
DATASET_REPOSITORY = "joyce8/EMBER2024"
DATASET_REVISION = "3d23efef7c0f0b702c5024400cfff4c3744a3832"
MODEL_REPOSITORY = "joyce8/EMBER2024-benchmark-models"
MODEL_REVISION = "e5b945dd90e1a1a1ec0cc07b3a17b52e9ba2d0c2"
MODEL_FILENAME = "EMBER2024_PE.model"
HISTORICAL_MODEL_SHA256 = "4252027863492ac138785c8c18576f43dad77d00faddc14e8c0072e8db419f99"

PE_TEST_ARCHIVES = {
    "Win32_test.zip": "Win32",
    "Win64_test.zip": "Win64",
    "Dot_Net_test.zip": "Dot_Net",
}

PE_TEST_ARCHIVE_INTEGRITY = {
    "Win32_test.zip": {
        "size_bytes": 2_593_425_203,
        "sha256": "c05f6562dee3ace4195087be918eb00181e33bc31464c671fb5ba00c9dd5dfdb",
    },
    "Win64_test.zip": {
        "size_bytes": 1_176_459_716,
        "sha256": "52a5a05c1bfa5bb021bb8b44c2e0afcf8983dfa1c6c0a9d76db393e5c682ce10",
    },
    "Dot_Net_test.zip": {
        "size_bytes": 220_481_493,
        "sha256": "b74c4181dbd77565fce16ba47c8ab0f7c7044ae6d880d859ec2c27365dea6299",
    },
}

MODEL_INTEGRITY = {
    "size_bytes": 3_756_042,
    "sha256": HISTORICAL_MODEL_SHA256,
}

EXTERNAL_ARTIFACTS = {
    filename: {
        "artifact_type": "dataset_archive",
        "repository": DATASET_REPOSITORY,
        "revision": DATASET_REVISION,
        "size_bytes": PE_TEST_ARCHIVE_INTEGRITY[filename]["size_bytes"],
        "sha256": PE_TEST_ARCHIVE_INTEGRITY[filename]["sha256"],
    }
    for filename in PE_TEST_ARCHIVES
}
EXTERNAL_ARTIFACTS[MODEL_FILENAME] = {
    "artifact_type": "benchmark_model",
    "repository": MODEL_REPOSITORY,
    "revision": MODEL_REVISION,
    "size_bytes": MODEL_INTEGRITY["size_bytes"],
    "sha256": MODEL_INTEGRITY["sha256"],
}
