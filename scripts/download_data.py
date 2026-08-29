#!/usr/bin/env python3
"""Opt-in interface for obtaining external EMBER2024 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.archive_manifest import (  # noqa: E402
    safely_extract_jsonl_archive,
    sha256_file,
    write_download_manifest,
)
from ember_calibration.upstream import (  # noqa: E402
    DATASET_REPOSITORY,
    DATASET_REVISION,
    HISTORICAL_MODEL_SHA256,
    MODEL_FILENAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PE_TEST_ARCHIVES,
    THREMBER_GIT_REVISION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="perform network downloads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"EMBER2024 PE destination: {args.data_dir.resolve()}")
    print(f"dataset source: {DATASET_REPOSITORY}@{DATASET_REVISION}")
    print(f"archives: {list(PE_TEST_ARCHIVES)}")
    print(f"model source: {MODEL_REPOSITORY}/{MODEL_FILENAME}@{MODEL_REVISION}")
    print(f"model destination: {args.model_dir.resolve()}")
    if not args.execute:
        print("dry run only; pass --execute after reviewing destinations and upstream revisions")
        return
    from huggingface_hub import hf_hub_download

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    archive_names = list(PE_TEST_ARCHIVES)
    if len(archive_names) != len(set(archive_names)):
        raise ValueError("duplicate archive names in configured download manifest")
    archive_records = []
    for archive_name, file_type in PE_TEST_ARCHIVES.items():
        downloaded = Path(
            hf_hub_download(
                repo_id=DATASET_REPOSITORY,
                repo_type="dataset",
                filename=archive_name,
                revision=DATASET_REVISION,
                local_dir=args.data_dir / "archives",
            )
        )
        extraction_dir = args.data_dir / "extracted" / downloaded.stem
        archive_records.append(
            safely_extract_jsonl_archive(downloaded, extraction_dir, file_type, args.data_dir)
        )
    write_download_manifest(args.data_dir / "download_manifest.json", archive_records, THREMBER_GIT_REVISION)
    model_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_dir=args.model_dir,
        )
    )
    model_checksum = sha256_file(model_path)
    if model_checksum != HISTORICAL_MODEL_SHA256:
        raise ValueError(
            "downloaded model does not match the historical SHA-256; stop and investigate: "
            f"{model_checksum}"
        )


if __name__ == "__main__":
    main()
