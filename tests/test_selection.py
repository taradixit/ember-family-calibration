import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import ember_calibration.selection as selection_module
from ember_calibration.selection import (
    DETECTOR_INPUT_FIELDS,
    EXPECTED_THREMBER_VERSION,
    SelectionError,
    load_completed_selection_manifest,
    plan_selection,
    select_records,
    sha256_file,
)
from ember_calibration.upstream import DATASET_REPOSITORY, DATASET_REVISION, PE_TEST_ARCHIVES
from scripts.select_test_records import parse_args

SYNTHETIC_DOCUMENTED_ROWS = {"Win32": 1, "Win64": 1, "Dot_Net": 1}
SYNTHETIC_COUNTS = {"Win32": 2, "Win64": 1, "Dot_Net": 1}


class FakeFeature:
    def __init__(self, name, dimension):
        self.name = name
        self.dim = dimension


class FakeExtractor:
    def __init__(self, fields=None, dimension=2568):
        names = fields or DETECTOR_INPUT_FIELDS
        dimensions = [7, 256, 256, 177, 74, 224, 1282, 129, 34, 33, 8, 88]
        self.features = [FakeFeature(name, size) for name, size in zip(names, dimensions)]
        self.dim = dimension


def synthetic_repeat_profile():
    return {
        "unique_hash_count": 3,
        "repeated_hash_count": 1,
        "repeated_hash_occurrences": 2,
        "multiplicity_summary": {1: 2, 2: 1},
        "repeated_hash_list_sha256": hashlib.sha256(b"repeat").hexdigest(),
        "conflict_counts": {
            "label": 0,
            "family": 0,
            "file_type": 0,
            "detector_input": 0,
        },
        "cross_member_and_week_repeated_hash_count": 1,
    }


def make_record(sample_hash, file_type, week_id, label, family):
    record = {
        "sha256": sample_hash,
        "label": label,
        "family": family,
        "file_type": file_type,
        "week_id": week_id,
        "caps": ["first"],
        "mbc": ["first"],
        "ttps": ["first"],
        "behavior": {"stable": True},
    }
    for field in DETECTOR_INPUT_FIELDS:
        record[field] = {"sample": sample_hash}
    return record


def default_pairs():
    first_records = {
        "win32-week-1.jsonl": make_record("repeat", "Win32", 1, 1, "family-a"),
        "win32-week-2.jsonl": make_record("repeat", "Win32", 2, 1, "family-a"),
        "win64-week-1.jsonl": make_record("win64", "Win64", 3, 0, None),
        "dotnet-week-1.jsonl": make_record("dotnet", "Dot_Net", 4, 1, "family-b"),
    }
    pairs = {}
    for name, first in first_records.items():
        second = deepcopy(first)
        second["caps"] = ["second"]
        second["mbc"] = ["second"]
        second["ttps"] = ["second"]
        pairs[name] = [first, second]
    return pairs


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def build_repository(tmp_path, change=None):
    pairs = default_pairs()
    if change is not None:
        change(pairs)
    names_by_archive = {
        "Win32_test.zip": ["win32-week-1.jsonl", "win32-week-2.jsonl"],
        "Win64_test.zip": ["win64-week-1.jsonl"],
        "Dot_Net_test.zip": ["dotnet-week-1.jsonl"],
    }
    raw_dir = tmp_path / "data/raw"
    archives = []
    for archive_name, file_type in PE_TEST_ARCHIVES.items():
        members = []
        for member_name in names_by_archive[archive_name]:
            relative = Path("extracted") / Path(archive_name).stem / member_name
            source = raw_dir / relative
            write_jsonl(source, pairs[member_name])
            members.append(
                {
                    "member_path": member_name,
                    "extracted_path": relative.as_posix(),
                    "sha256": sha256_file(source),
                    "size_bytes": source.stat().st_size,
                    "is_jsonl": True,
                }
            )
        archives.append(
            {
                "archive_filename": archive_name,
                "assigned_file_type": file_type,
                "members": members,
            }
        )
    manifest = raw_dir / "download_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_repository": DATASET_REPOSITORY,
                "dataset_revision": DATASET_REVISION,
                "archives": archives,
            }
        )
    )
    return manifest, tmp_path / "data/selected", pairs


def run_selection(tmp_path, change=None, extractor=None, repeat_profile=None, overwrite=False):
    manifest, output_dir, pairs = build_repository(tmp_path, change)
    result = select_records(
        manifest,
        output_dir,
        tmp_path,
        extractor or FakeExtractor(),
        EXPECTED_THREMBER_VERSION,
        overwrite=overwrite,
        documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
        expected_counts=SYNTHETIC_COUNTS,
        expected_repeat_profile=repeat_profile or synthetic_repeat_profile(),
        creation_time="2026-01-01T00:00:00Z",
    )
    return result, manifest, output_dir, pairs


def test_successful_selection_preserves_member_order_counts_and_first_half(tmp_path):
    selection_manifest, source_manifest, _, pairs = run_selection(tmp_path)
    document = json.loads(selection_manifest.read_text())

    assert [item["member_name"] for item in document["selected_member_order"]] == [
        "win32-week-1.jsonl",
        "win32-week-2.jsonl",
        "win64-week-1.jsonl",
        "dotnet-week-1.jsonl",
    ]
    assert document["total_selected_counts_by_file_type"] == SYNTHETIC_COUNTS
    assert document["total_selected_row_count"] == 4
    assert document["completion_status"] == "complete"
    assert document["detector_input_fields"] == DETECTOR_INPUT_FIELDS
    assert document["extractor_dimension"] == 2568
    assert document["allowed_metadata_only_differences"] == ["caps", "mbc", "ttps"]
    assert document["source_manifest"]["sha256"] == sha256_file(source_manifest)
    for item in document["selected_member_order"]:
        assert not Path(item["source_member_path"]).is_absolute()
        assert not Path(item["selected_output_path"]).is_absolute()
        output_path = tmp_path / item["selected_output_path"]
        selected = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert selected == [pairs[item["member_name"]][0]]
        assert item["selected_output_sha256"] == sha256_file(output_path)


@pytest.mark.parametrize("allowed_field", ["caps", "mbc", "ttps"])
def test_each_reviewed_metadata_difference_is_allowed(tmp_path, allowed_field):
    def change(pairs):
        for records in pairs.values():
            records[1]["caps"] = records[0]["caps"]
            records[1]["mbc"] = records[0]["mbc"]
            records[1]["ttps"] = records[0]["ttps"]
            records[1][allowed_field] = ["different"]

    selection_manifest, _, _, _ = run_selection(tmp_path, change)
    assert selection_manifest.is_file()


def test_unreviewed_field_difference_is_rejected(tmp_path):
    def change(pairs):
        pairs["win32-week-1.jsonl"][1]["behavior"] = {"stable": False}

    with pytest.raises(SelectionError, match="unapproved paired field difference"):
        run_selection(tmp_path, change)


def test_source_sha_mismatch_is_rejected(tmp_path):
    manifest, output_dir, _ = build_repository(tmp_path)
    document = json.loads(manifest.read_text())
    document["archives"][0]["members"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(SelectionError, match="source SHA-256 mismatch"):
        select_records(
            manifest,
            output_dir,
            tmp_path,
            FakeExtractor(),
            EXPECTED_THREMBER_VERSION,
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
        )


def test_source_size_mismatch_is_rejected(tmp_path):
    manifest, output_dir, _ = build_repository(tmp_path)
    document = json.loads(manifest.read_text())
    document["archives"][0]["members"][0]["size_bytes"] += 1
    manifest.write_text(json.dumps(document))
    with pytest.raises(SelectionError, match="source size mismatch"):
        select_records(
            manifest,
            output_dir,
            tmp_path,
            FakeExtractor(),
            EXPECTED_THREMBER_VERSION,
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
        )


def test_odd_or_incorrect_source_row_count_is_rejected(tmp_path):
    manifest, output_dir, pairs = build_repository(tmp_path)
    document = json.loads(manifest.read_text())
    member = document["archives"][0]["members"][0]
    source = tmp_path / "data/raw" / member["extracted_path"]
    write_jsonl(source, pairs[member["member_path"]] + [pairs[member["member_path"]][0]])
    member["size_bytes"] = source.stat().st_size
    member["sha256"] = sha256_file(source)
    manifest.write_text(json.dumps(document))
    with pytest.raises(SelectionError, match="source row count mismatch"):
        select_records(
            manifest,
            output_dir,
            tmp_path,
            FakeExtractor(),
            EXPECTED_THREMBER_VERSION,
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
        )


@pytest.mark.parametrize(
    ("field", "new_value", "message"),
    [
        ("sha256", "other", "paired sha256 mismatch"),
        ("label", 0, "paired label mismatch"),
        ("family", "other-family", "paired family mismatch"),
        ("file_type", "Win64", "paired file_type mismatch"),
        ("week_id", 99, "paired week_id mismatch"),
    ],
)
def test_paired_evaluation_metadata_mismatch_is_rejected(tmp_path, field, new_value, message):
    def change(pairs):
        pairs["win32-week-1.jsonl"][1][field] = new_value

    with pytest.raises(SelectionError, match=message):
        run_selection(tmp_path, change)


def test_paired_detector_input_mismatch_is_rejected(tmp_path):
    def change(pairs):
        pairs["win32-week-1.jsonl"][1]["general"] = {"changed": True}

    with pytest.raises(SelectionError, match="paired detector-input mismatch"):
        run_selection(tmp_path, change)


@pytest.mark.parametrize("missing_field", ["sha256", "general"])
def test_missing_required_key_is_rejected(tmp_path, missing_field):
    def change(pairs):
        del pairs["win32-week-1.jsonl"][1][missing_field]

    with pytest.raises(SelectionError, match="missing required fields"):
        run_selection(tmp_path, change)


def test_incorrect_extractor_field_mapping_is_rejected(tmp_path):
    fields = list(DETECTOR_INPUT_FIELDS)
    fields[-1] = "wrong"
    with pytest.raises(SelectionError, match="extractor field mapping mismatch"):
        run_selection(tmp_path, extractor=FakeExtractor(fields=fields))


def test_incorrect_extractor_dimension_is_rejected(tmp_path):
    with pytest.raises(SelectionError, match="extractor dimension mismatch"):
        run_selection(tmp_path, extractor=FakeExtractor(dimension=999))


def test_failed_write_removes_all_temporary_and_selected_outputs(tmp_path):
    def change(pairs):
        pairs["win64-week-1.jsonl"][1]["behavior"] = {"stable": False}

    with pytest.raises(SelectionError, match="unapproved paired field difference"):
        run_selection(tmp_path, change)
    output_dir = tmp_path / "data/selected"
    assert not list(output_dir.rglob("*.tmp"))
    assert not list(output_dir.rglob("*.jsonl"))
    assert not (output_dir / "selection_manifest.json").exists()
    assert not list(output_dir.parent.glob(".selected.staging-*"))


def test_existing_completed_selection_requires_explicit_overwrite(tmp_path):
    _, manifest, output_dir, _ = run_selection(tmp_path)
    with pytest.raises(SelectionError, match="pass --overwrite"):
        select_records(
            manifest,
            output_dir,
            tmp_path,
            FakeExtractor(),
            EXPECTED_THREMBER_VERSION,
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
        )


def test_completed_selection_revalidates_outputs_and_repeat_profile(tmp_path):
    selection_manifest, _, _, _ = run_selection(tmp_path)
    document, inputs = load_completed_selection_manifest(
        selection_manifest,
        tmp_path,
        FakeExtractor(),
        expected_counts=SYNTHETIC_COUNTS,
        expected_repeat_profile=synthetic_repeat_profile(),
        documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
    )
    assert document["repeated_hash_count"] == 1
    assert [file_type for file_type, _ in inputs] == ["Win32", "Win32", "Win64", "Dot_Net"]


def test_selected_output_checksum_mismatch_is_rejected(tmp_path):
    selection_manifest, _, _, _ = run_selection(tmp_path)
    document = json.loads(selection_manifest.read_text())
    selected = tmp_path / document["selected_member_order"][0]["selected_output_path"]
    content = selected.read_bytes()
    selected.write_bytes(b"x" + content[1:])
    with pytest.raises(SelectionError, match="selected output SHA-256 mismatch"):
        load_completed_selection_manifest(
            selection_manifest,
            tmp_path,
            FakeExtractor(),
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
        )


def test_incomplete_or_wrong_total_selection_manifest_is_rejected(tmp_path):
    selection_manifest, _, _, _ = run_selection(tmp_path)
    document = json.loads(selection_manifest.read_text())
    document["completion_status"] = "incomplete"
    selection_manifest.write_text(json.dumps(document))
    with pytest.raises(SelectionError, match="incomplete"):
        load_completed_selection_manifest(
            selection_manifest,
            tmp_path,
            FakeExtractor(),
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
        )

    document["completion_status"] = "complete"
    document["total_selected_row_count"] = 5
    selection_manifest.write_text(json.dumps(document))
    with pytest.raises(SelectionError, match="wrong total row count"):
        load_completed_selection_manifest(
            selection_manifest,
            tmp_path,
            FakeExtractor(),
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
        )


def test_recorded_residual_repeat_profile_is_revalidated(tmp_path):
    selection_manifest, _, _, _ = run_selection(tmp_path)
    document = json.loads(selection_manifest.read_text())
    document["repeated_hash_list_sha256"] = "0" * 64
    selection_manifest.write_text(json.dumps(document))
    with pytest.raises(SelectionError, match="repeated_hash_list_sha256"):
        load_completed_selection_manifest(
            selection_manifest,
            tmp_path,
            FakeExtractor(),
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
        )


def test_unexpected_repeated_hash_is_rejected(tmp_path):
    def change(pairs):
        for record in pairs["win32-week-2.jsonl"]:
            record["sha256"] = "unexpected"
            for field in DETECTOR_INPUT_FIELDS:
                record[field] = {"sample": "unexpected"}

    with pytest.raises(SelectionError, match="unique_hash_count"):
        run_selection(tmp_path, change)


@pytest.mark.parametrize(
    ("field", "new_value", "message"),
    [
        ("label", 0, "conflict_counts"),
        ("family", "other-family", "conflict_counts"),
        ("general", {"changed": True}, "conflict_counts"),
    ],
)
def test_repeated_hash_conflicts_are_rejected(tmp_path, field, new_value, message):
    def change(pairs):
        for record in pairs["win32-week-2.jsonl"]:
            record[field] = new_value

    with pytest.raises(SelectionError, match=message):
        run_selection(tmp_path, change)


def test_repeated_hash_file_type_conflict_is_rejected(tmp_path):
    def change(pairs):
        for record in pairs["win64-week-1.jsonl"]:
            record["sha256"] = "repeat"
            record["label"] = 1
            record["family"] = "family-a"
            for field in DETECTOR_INPUT_FIELDS:
                record[field] = {"sample": "repeat"}

    expected_profile = {
        "unique_hash_count": 2,
        "repeated_hash_count": 1,
        "repeated_hash_occurrences": 3,
        "multiplicity_summary": {1: 1, 3: 1},
        "repeated_hash_list_sha256": hashlib.sha256(b"repeat").hexdigest(),
        "conflict_counts": {
            "label": 0,
            "family": 0,
            "file_type": 0,
            "detector_input": 0,
        },
        "cross_member_and_week_repeated_hash_count": 1,
    }
    with pytest.raises(SelectionError, match="conflict_counts"):
        run_selection(tmp_path, change, repeat_profile=expected_profile)


def test_dry_run_plan_does_not_read_or_write_selected_files(tmp_path):
    manifest, output_dir, _ = build_repository(tmp_path)
    for source in (tmp_path / "data/raw/extracted").rglob("*.jsonl"):
        source.unlink()
    plan = plan_selection(
        manifest,
        output_dir,
        tmp_path,
        documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
        expected_counts=SYNTHETIC_COUNTS,
    )
    assert plan["total_selected_rows"] == 4
    assert not output_dir.exists()


def test_overwrite_requires_execute_argument():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--download-manifest",
                "data/raw/download_manifest.json",
                "--output-dir",
                "data/selected",
                "--overwrite",
            ]
        )


def test_preparation_loader_refuses_original_download_manifest(tmp_path):
    manifest, _, _ = build_repository(tmp_path)
    with pytest.raises(SelectionError, match="wrong selection rule"):
        load_completed_selection_manifest(
            manifest,
            tmp_path,
            FakeExtractor(),
            expected_counts=SYNTHETIC_COUNTS,
            expected_repeat_profile=synthetic_repeat_profile(),
            documented_rows=SYNTHETIC_DOCUMENTED_ROWS,
        )


def test_failure_immediately_before_publication_leaves_no_final_selection(
    tmp_path, monkeypatch
):
    def fail_before_publication(staging_dir, selected_members):
        raise SelectionError("failure immediately before publication")

    monkeypatch.setattr(
        selection_module,
        "validate_staged_publication",
        fail_before_publication,
    )
    with pytest.raises(SelectionError, match="immediately before publication"):
        run_selection(tmp_path)
    output_dir = tmp_path / "data/selected"
    assert not output_dir.exists()
    assert not list(output_dir.parent.glob(".selected.staging-*"))
    assert not list(output_dir.parent.glob(".selected.backup-*"))


def test_simulated_first_publication_failure_leaves_no_final_selection(
    tmp_path, monkeypatch
):
    original_rename = selection_module.rename_directory

    def fail_staging_publish(source, destination):
        if source.name.startswith(".selected.staging-") and destination.name == "selected":
            raise OSError("simulated publication failure")
        original_rename(source, destination)

    monkeypatch.setattr(selection_module, "rename_directory", fail_staging_publish)
    with pytest.raises(OSError, match="simulated publication failure"):
        run_selection(tmp_path)
    output_dir = tmp_path / "data/selected"
    assert not output_dir.exists()
    assert not list(output_dir.parent.glob(".selected.staging-*"))
    assert not list(output_dir.parent.glob(".selected.backup-*"))


def test_overwrite_publication_failure_restores_previous_complete_selection(
    tmp_path, monkeypatch
):
    _, _, output_dir, _ = run_selection(tmp_path)
    previous_files = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    original_rename = selection_module.rename_directory

    def fail_replacement_publish(source, destination):
        if source.name.startswith(".selected.staging-") and destination == output_dir:
            raise OSError("simulated replacement publication failure")
        original_rename(source, destination)

    monkeypatch.setattr(selection_module, "rename_directory", fail_replacement_publish)
    with pytest.raises(OSError, match="replacement publication failure"):
        run_selection(tmp_path, overwrite=True)
    restored_files = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert restored_files == previous_files
    assert not list(output_dir.parent.glob(".selected.staging-*"))
    assert not list(output_dir.parent.glob(".selected.backup-*"))


def test_successful_directory_publication_is_complete_and_leaves_no_work_dirs(tmp_path):
    selection_manifest, _, output_dir, _ = run_selection(tmp_path)
    document = json.loads(selection_manifest.read_text())
    expected_outputs = {
        tmp_path / member["selected_output_path"]
        for member in document["selected_member_order"]
    }
    assert selection_manifest == output_dir / "selection_manifest.json"
    assert all(path.is_file() for path in expected_outputs)
    assert len(expected_outputs) == len(document["selected_member_order"])
    assert not list(output_dir.parent.glob(".selected.staging-*"))
    assert not list(output_dir.parent.glob(".selected.backup-*"))
