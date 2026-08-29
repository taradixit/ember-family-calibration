import json
import zipfile

import pytest

from ember_calibration.archive_manifest import (
    load_jsonl_inputs,
    safely_extract_jsonl_archive,
    write_download_manifest,
)
from ember_calibration.upstream import PE_TEST_ARCHIVES, THREMBER_GIT_REVISION


def write_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)


def build_valid_manifest(tmp_path):
    records = []
    contents = {
        "Win32_test.zip": [("part-1.jsonl", "{}\n"), ("nested/part-2.jsonl", "{}\n")],
        "Win64_test.zip": [("win64.jsonl", "{}\n")],
        "Dot_Net_test.zip": [("dotnet.jsonl", "{}\n")],
    }
    for archive_name, file_type in PE_TEST_ARCHIVES.items():
        archive_path = tmp_path / archive_name
        write_zip(archive_path, contents[archive_name])
        records.append(
            safely_extract_jsonl_archive(
                archive_path,
                tmp_path / "extracted" / archive_path.stem,
                file_type,
                tmp_path,
            )
        )
    manifest = tmp_path / "download_manifest.json"
    write_download_manifest(manifest, records, THREMBER_GIT_REVISION)
    return manifest


def test_manifest_preserves_all_jsonl_members_in_recorded_order(tmp_path):
    manifest = build_valid_manifest(tmp_path)
    inputs = load_jsonl_inputs(manifest)
    assert [file_type for file_type, _ in inputs] == ["Win32", "Win32", "Win64", "Dot_Net"]
    assert [path.name for _, path in inputs] == [
        "part-1.jsonl",
        "part-2.jsonl",
        "win64.jsonl",
        "dotnet.jsonl",
    ]


@pytest.mark.parametrize("unsafe_name", ["../escape.jsonl", "/absolute.jsonl", "dir\\escape.jsonl"])
def test_safe_extraction_rejects_path_traversal(tmp_path, unsafe_name):
    archive_path = tmp_path / "Win32_test.zip"
    write_zip(archive_path, [(unsafe_name, "{}\n")])
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        safely_extract_jsonl_archive(archive_path, tmp_path / "out", "Win32", tmp_path)
    assert not (tmp_path.parent / "escape.jsonl").exists()


def test_safe_extraction_rejects_unexpected_members(tmp_path):
    archive_path = tmp_path / "Win32_test.zip"
    write_zip(archive_path, [("README.txt", "unexpected")])
    with pytest.raises(ValueError, match="unexpected archive member"):
        safely_extract_jsonl_archive(archive_path, tmp_path / "out", "Win32", tmp_path)


def test_safe_extraction_rejects_empty_jsonl_archive(tmp_path):
    archive_path = tmp_path / "Win32_test.zip"
    write_zip(archive_path, [("folder/", "")])
    with pytest.raises(ValueError, match="no JSONL"):
        safely_extract_jsonl_archive(archive_path, tmp_path / "out", "Win32", tmp_path)


def test_safe_extraction_rejects_duplicate_member_paths(tmp_path):
    archive_path = tmp_path / "Win32_test.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("same.jsonl", "{}\n")
            archive.writestr("same.jsonl", "{}\n")
    with pytest.raises(ValueError, match="duplicate ZIP member"):
        safely_extract_jsonl_archive(archive_path, tmp_path / "out", "Win32", tmp_path)


def test_manifest_rejects_duplicate_archive_names(tmp_path):
    manifest = build_valid_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["archives"].append(document["archives"][0])
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate archive name"):
        load_jsonl_inputs(manifest)


def test_manifest_rejects_duplicate_recorded_member_paths(tmp_path):
    manifest = build_valid_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["archives"][0]["members"].append(document["archives"][0]["members"][0])
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate member path"):
        load_jsonl_inputs(manifest)


def test_manifest_rejects_duplicate_resolved_input_paths(tmp_path):
    manifest = build_valid_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    first = document["archives"][0]["members"][0]
    second = document["archives"][1]["members"][0]
    second["extracted_path"] = first["extracted_path"]
    second["size_bytes"] = first["size_bytes"]
    second["sha256"] = first["sha256"]
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate resolved input"):
        load_jsonl_inputs(manifest)


def test_manifest_rejects_an_empty_jsonl_list(tmp_path):
    manifest = build_valid_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    for archive in document["archives"]:
        archive["members"] = []
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="empty JSONL"):
        load_jsonl_inputs(manifest)
