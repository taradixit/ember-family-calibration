"""Safe ZIP extraction and explicit archive-manifest validation."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from .upstream import DATASET_REPOSITORY, DATASET_REVISION, PE_TEST_ARCHIVES, THREMBER_GIT_REVISION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def safely_extract_jsonl_archive(
    archive_path: Path, output_dir: Path, assigned_file_type: str, manifest_base: Path
) -> dict[str, object]:
    """Extract only regular JSONL members and return their ordered manifest."""
    members: list[dict[str, object]] = []
    seen_names: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.filename in seen_names:
                raise ValueError(f"duplicate ZIP member path: {info.filename}")
            seen_names.add(info.filename)
            relative_member = _safe_member_path(info.filename)
            if info.is_dir():
                continue
            if _is_symlink(info) or relative_member.suffix.lower() != ".jsonl":
                raise ValueError(f"unexpected archive member: {info.filename}")
            destination = output_dir.joinpath(*relative_member.parts)
            resolved_destination = destination.resolve()
            if not resolved_destination.is_relative_to(output_dir.resolve()):
                raise ValueError(f"unsafe ZIP extraction target: {info.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            members.append(
                {
                    "member_path": info.filename,
                    "extracted_path": str(resolved_destination.relative_to(manifest_base.resolve())),
                    "sha256": sha256_file(destination),
                    "size_bytes": destination.stat().st_size,
                    "is_jsonl": True,
                }
            )
    if not members:
        raise ValueError(f"archive contains no JSONL inputs: {archive_path.name}")
    return {
        "archive_filename": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "assigned_file_type": assigned_file_type,
        "members": members,
    }


def write_download_manifest(path: Path, archives: list[dict[str, object]], thrember_revision: str) -> None:
    document = {
        "schema_version": 1,
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "thrember_git_revision": thrember_revision,
        "archives": archives,
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def load_jsonl_inputs(manifest_path: Path) -> list[tuple[str, Path]]:
    """Validate a download manifest and return JSONLs in recorded order."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("dataset_repository") != DATASET_REPOSITORY:
        raise ValueError("unexpected dataset repository in download manifest")
    if document.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("unexpected dataset revision in download manifest")
    if document.get("thrember_git_revision") != THREMBER_GIT_REVISION:
        raise ValueError("unexpected thrember Git revision in download manifest")
    archives = document.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError("download manifest has no archives")
    archive_names: set[str] = set()
    extracted_paths: set[Path] = set()
    inputs: list[tuple[str, Path]] = []
    for archive in archives:
        name = archive.get("archive_filename")
        if name in archive_names:
            raise ValueError(f"duplicate archive name: {name}")
        archive_names.add(name)
        if name not in PE_TEST_ARCHIVES:
            raise ValueError(f"unexpected archive name: {name}")
        file_type = archive.get("assigned_file_type")
        if file_type != PE_TEST_ARCHIVES[name]:
            raise ValueError(f"incorrect file type for archive {name}")
        members = archive.get("members")
        if not isinstance(members, list):
            raise ValueError(f"archive members must be a list: {name}")
        member_names: set[str] = set()
        for member in members:
            member_name = member.get("member_path")
            if member_name in member_names:
                raise ValueError(f"duplicate member path in {name}: {member_name}")
            member_names.add(member_name)
            _safe_member_path(str(member_name))
            if member.get("is_jsonl") is not True or not str(member_name).lower().endswith(".jsonl"):
                raise ValueError(f"unexpected archive member: {member_name}")
            relative_path = _safe_member_path(str(member.get("extracted_path")))
            resolved = manifest_path.parent.joinpath(*relative_path.parts).resolve()
            if not resolved.is_relative_to(manifest_path.parent.resolve()):
                raise ValueError(f"unsafe extracted member path: {relative_path}")
            if resolved in extracted_paths:
                raise ValueError(f"duplicate resolved input path: {resolved}")
            extracted_paths.add(resolved)
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            if resolved.stat().st_size != member.get("size_bytes"):
                raise ValueError(f"member size mismatch: {resolved}")
            if sha256_file(resolved) != member.get("sha256"):
                raise ValueError(f"member checksum mismatch: {resolved}")
            inputs.append((str(file_type), resolved))
    if archive_names != set(PE_TEST_ARCHIVES):
        raise ValueError(f"archive set mismatch: found {sorted(archive_names)}")
    if not inputs:
        raise ValueError("download manifest has an empty JSONL input list")
    return inputs
