#!/usr/bin/env python3
"""Download and verify pinned EMBER2024 artifacts before any extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.archive_manifest import (  # noqa: E402
    safely_extract_jsonl_archive,
    write_download_manifest,
)
from ember_calibration.artifact_integrity import verify_external_artifact  # noqa: E402
from ember_calibration.upstream import (  # noqa: E402
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EXTERNAL_ARTIFACTS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PE_TEST_ARCHIVES,
    THREMBER_GIT_REVISION,
)

DownloadFunction = Callable[..., str]
ExtractFunction = Callable[[Path, Path, str, Path], dict[str, object]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="perform network downloads")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download and verify artifacts without opening or extracting ZIP files",
    )
    args = parser.parse_args(argv)
    if args.download_only and not args.execute:
        parser.error("--download-only requires --execute")
    return args


def repository_relative_path(path: Path, repository_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError("artifact destinations must be inside the repository") from error
    return relative.as_posix()


def artifact_destination(
    filename: str, artifact: dict[str, object], data_dir: Path, model_dir: Path
) -> Path:
    if artifact["artifact_type"] == "dataset_archive":
        return data_dir / "archives" / filename
    return model_dir / filename


def print_dry_run(args: argparse.Namespace, repository_root: Path) -> None:
    print("dry run: no network requests will be made")
    for filename, artifact in EXTERNAL_ARTIFACTS.items():
        destination = artifact_destination(filename, artifact, args.data_dir, args.model_dir)
        local_path = repository_relative_path(destination, repository_root)
        print(f"- {filename}")
        print(f"  repository: {artifact['repository']}")
        print(f"  revision: {artifact['revision']}")
        print(f"  size bytes: {artifact['size_bytes']}")
        print(f"  sha256: {artifact['sha256']}")
        print(f"  destination: {local_path}")


def download_artifacts(
    data_dir: Path,
    model_dir: Path,
    download_function: DownloadFunction,
    expected_artifacts: dict[str, dict[str, object]],
) -> dict[str, Path]:
    downloaded: dict[str, Path] = {}
    for filename, artifact in expected_artifacts.items():
        local_dir = data_dir / "archives"
        arguments: dict[str, object] = {
            "repo_id": artifact["repository"],
            "filename": filename,
            "revision": artifact["revision"],
        }
        if artifact["artifact_type"] == "dataset_archive":
            arguments["repo_type"] = "dataset"
        else:
            local_dir = model_dir
        arguments["local_dir"] = local_dir
        downloaded[filename] = Path(download_function(**arguments))
    return downloaded


def verify_downloads(
    downloaded: dict[str, Path],
    expected_artifacts: dict[str, dict[str, object]],
    repository_root: Path,
) -> list[dict[str, object]]:
    records = []
    for filename, artifact in expected_artifacts.items():
        verification = verify_external_artifact(
            downloaded[filename],
            filename,
            int(artifact["size_bytes"]),
            str(artifact["sha256"]),
        )
        verification.update(
            {
                "artifact_type": artifact["artifact_type"],
                "repository": artifact["repository"],
                "revision": artifact["revision"],
                "local_path": repository_relative_path(downloaded[filename], repository_root),
                "extraction_occurred": False,
            }
        )
        records.append(verification)
    return records


def write_external_artifact_manifest(path: Path, records: list[dict[str, object]]) -> None:
    document = {
        "schema_version": 1,
        "repositories": {
            "dataset": {"identifier": DATASET_REPOSITORY, "revision": DATASET_REVISION},
            "benchmark_model": {"identifier": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        },
        "artifacts": records,
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def run_download(
    args: argparse.Namespace,
    download_function: DownloadFunction,
    extract_function: ExtractFunction = safely_extract_jsonl_archive,
    repository_root: Path | None = None,
    expected_artifacts: dict[str, dict[str, object]] = EXTERNAL_ARTIFACTS,
    archive_mapping: dict[str, str] = PE_TEST_ARCHIVES,
) -> Path:
    root = repository_root or Path(__file__).resolve().parents[1]
    repository_relative_path(args.data_dir, root)
    repository_relative_path(args.model_dir, root)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    external_manifest = args.data_dir / "external_artifact_manifest.json"
    extraction_manifest = args.data_dir / "download_manifest.json"
    external_manifest.unlink(missing_ok=True)
    if not args.download_only:
        extraction_manifest.unlink(missing_ok=True)

    downloaded = download_artifacts(
        args.data_dir, args.model_dir, download_function, expected_artifacts
    )
    records = verify_downloads(downloaded, expected_artifacts, root)
    write_external_artifact_manifest(external_manifest, records)
    if args.download_only:
        return external_manifest

    records_by_name = {str(record["filename"]): record for record in records}
    archive_records = []
    for filename, file_type in archive_mapping.items():
        downloaded_path = downloaded[filename]
        extraction_dir = args.data_dir / "extracted" / downloaded_path.stem
        archive_records.append(
            extract_function(downloaded_path, extraction_dir, file_type, args.data_dir)
        )
        records_by_name[filename]["extraction_occurred"] = True
        write_external_artifact_manifest(external_manifest, records)
    write_download_manifest(extraction_manifest, archive_records, THREMBER_GIT_REVISION)
    return external_manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    if not args.execute:
        print_dry_run(args, repository_root)
        return
    from huggingface_hub import hf_hub_download

    manifest = run_download(args, hf_hub_download, repository_root=repository_root)
    print(f"verified artifact manifest: {repository_relative_path(manifest, repository_root)}")


if __name__ == "__main__":
    main()
