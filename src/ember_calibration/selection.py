"""Reviewed first-half selection for the released EMBER2024 PE test data."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .upstream import DATASET_REPOSITORY, DATASET_REVISION, PE_TEST_ARCHIVES

DOCUMENTED_ROWS_PER_MEMBER = {"Win32": 30_000, "Win64": 10_000, "Dot_Net": 5_000}
EXPECTED_SELECTED_COUNTS = {"Win32": 360_000, "Win64": 120_000, "Dot_Net": 60_000}
EXPECTED_SELECTED_ROWS = sum(EXPECTED_SELECTED_COUNTS.values())
EXPECTED_EXTRACTOR_DIMENSION = 2568
EXPECTED_THREMBER_VERSION = "0.1.0"
DETECTOR_INPUT_FIELDS = [
    "general",
    "histogram",
    "byteentropy",
    "strings",
    "header",
    "section",
    "imports",
    "exports",
    "datadirectories",
    "richheader",
    "authenticode",
    "pefilewarnings",
]
ALLOWED_METADATA_DIFFERENCES = ["caps", "mbc", "ttps"]
SELECTION_RULE = "first_documented_half_after_pairwise_equivalence"
REPEATED_HASH_LIST_SHA256 = "81c20f8d9397f4f27143652988dfdc036edc3a3c948a0efe75e7817e97283767"
REVIEWED_REPEAT_PROFILE = {
    "unique_hash_count": 539_940,
    "repeated_hash_count": 60,
    "repeated_hash_occurrences": 120,
    "multiplicity_summary": {1: 539_880, 2: 60},
    "repeated_hash_list_sha256": REPEATED_HASH_LIST_SHA256,
    "conflict_counts": {
        "label": 0,
        "family": 0,
        "file_type": 0,
        "detector_input": 0,
    },
    "cross_member_and_week_repeated_hash_count": 60,
}
REQUIRED_PAIR_FIELDS = {"sha256", "label", "family", "file_type", "week_id"}
HASH_CHUNK_SIZE = 1024 * 1024


class SelectionError(ValueError):
    """Raised when source data does not satisfy the reviewed selection policy."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise SelectionError("selection paths must remain inside the repository") from error


def resolve_repository_path(value: object, repository_root: Path) -> Path:
    relative = PurePosixPath(str(value))
    if not str(value) or relative.is_absolute() or ".." in relative.parts or "\\" in str(value):
        raise SelectionError(f"unsafe repository-relative path: {value}")
    resolved = repository_root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(repository_root.resolve()):
        raise SelectionError(f"path escapes repository: {value}")
    return resolved


def _operation_prefix(output_dir: Path, operation: str) -> str:
    return f".{output_dir.name}.{operation}-"


def create_staging_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=_operation_prefix(output_dir, "staging"),
            dir=output_dir.parent,
        )
    )


def reserve_backup_path(output_dir: Path) -> Path:
    backup_path = Path(
        tempfile.mkdtemp(
            prefix=_operation_prefix(output_dir, "backup"),
            dir=output_dir.parent,
        )
    )
    backup_path.rmdir()
    return backup_path


def remove_operation_directory(path: Path, output_dir: Path, operation: str) -> None:
    if not path.exists():
        return
    expected_parent = output_dir.parent.resolve()
    if path.parent.resolve() != expected_parent or not path.name.startswith(
        _operation_prefix(output_dir, operation)
    ):
        raise SelectionError(f"refusing to remove unexpected {operation} directory: {path}")
    shutil.rmtree(path)


def rename_directory(source: Path, destination: Path) -> None:
    source.replace(destination)


def validate_staged_publication(
    staging_dir: Path,
    selected_members: list[dict[str, object]],
) -> None:
    if not (staging_dir / "selection_manifest.json").is_file():
        raise SelectionError("staged selection manifest is missing before publication")
    for member in selected_members:
        relative_to_selection = Path(Path(str(member["archive_name"])).stem) / Path(
            str(member["member_name"])
        )
        if not (staging_dir / relative_to_selection).is_file():
            raise SelectionError(
                f"staged selected member is missing before publication: {relative_to_selection}"
            )


def publish_staged_selection(staging_dir: Path, output_dir: Path, overwrite: bool) -> None:
    backup_path: Path | None = None
    previous_moved = False
    if output_dir.exists():
        if not overwrite:
            raise SelectionError("completed selection already exists; pass --overwrite to replace it")
        backup_path = reserve_backup_path(output_dir)
        rename_directory(output_dir, backup_path)
        previous_moved = True
    try:
        rename_directory(staging_dir, output_dir)
    except Exception:
        if output_dir.exists() and not staging_dir.exists():
            rename_directory(output_dir, staging_dir)
        if previous_moved and backup_path is not None and not output_dir.exists():
            rename_directory(backup_path, output_dir)
        raise
    if backup_path is not None:
        remove_operation_directory(backup_path, output_dir, "backup")


def canonical_token(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def detector_projection_digest(record: dict[str, object], fields: list[str]) -> str:
    projection = [[field, record[field]] for field in fields]
    encoded = json.dumps(
        projection,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_extractor_mapping(
    extractor: object,
    expected_fields: list[str] = DETECTOR_INPUT_FIELDS,
    expected_dimension: int = EXPECTED_EXTRACTOR_DIMENSION,
) -> dict[str, object]:
    features = getattr(extractor, "features", None)
    if not isinstance(features, list) or not features:
        raise SelectionError("installed extractor has no inspectable feature components")
    component_records = []
    observed_fields = []
    component_dimension = 0
    for feature in features:
        name = getattr(feature, "name", None)
        dimension = getattr(feature, "dim", None)
        if not isinstance(name, str) or not name:
            raise SelectionError("installed extractor has an unnamed feature component")
        if not isinstance(dimension, int) or dimension < 1:
            raise SelectionError(f"extractor component has an invalid dimension: {name}")
        observed_fields.append(name)
        component_dimension += dimension
        component_records.append(
            {
                "component_name": type(feature).__name__,
                "top_level_json_field": name,
                "dimension": dimension,
            }
        )
    if observed_fields != expected_fields:
        raise SelectionError(
            f"extractor field mapping mismatch: expected {expected_fields}, found {observed_fields}"
        )
    extractor_dimension = getattr(extractor, "dim", None)
    if extractor_dimension != expected_dimension or component_dimension != expected_dimension:
        raise SelectionError(
            "extractor dimension mismatch: "
            f"expected {expected_dimension}, found extractor={extractor_dimension}, "
            f"components={component_dimension}"
        )
    return {
        "dimension": extractor_dimension,
        "detector_input_fields": observed_fields,
        "feature_components": component_records,
    }


def read_source_manifest(
    manifest_path: Path,
    output_dir: Path,
    repository_root: Path,
    documented_rows: dict[str, int] = DOCUMENTED_ROWS_PER_MEMBER,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("dataset_repository") != DATASET_REPOSITORY:
        raise SelectionError("unexpected dataset repository in download manifest")
    if document.get("dataset_revision") != DATASET_REVISION:
        raise SelectionError("unexpected dataset revision in download manifest")
    archives = document.get("archives")
    if not isinstance(archives, list) or not archives:
        raise SelectionError("download manifest has no archives")
    archive_names = set()
    source_paths = set()
    selected_paths = set()
    members = []
    manifest_base = manifest_path.parent.resolve()
    repository_relative_path(output_dir, repository_root)
    for archive in archives:
        archive_name = archive.get("archive_filename")
        if archive_name in archive_names:
            raise SelectionError(f"duplicate archive name: {archive_name}")
        archive_names.add(archive_name)
        if archive_name not in PE_TEST_ARCHIVES:
            raise SelectionError(f"unexpected archive name: {archive_name}")
        file_type = archive.get("assigned_file_type")
        if file_type != PE_TEST_ARCHIVES[archive_name] or file_type not in documented_rows:
            raise SelectionError(f"incorrect file type for archive: {archive_name}")
        archive_members = archive.get("members")
        if not isinstance(archive_members, list) or not archive_members:
            raise SelectionError(f"archive has no JSONL members: {archive_name}")
        member_names = set()
        for member in archive_members:
            member_name = PurePosixPath(str(member.get("member_path")))
            if (
                not str(member.get("member_path"))
                or member_name.is_absolute()
                or ".." in member_name.parts
                or "\\" in str(member.get("member_path"))
                or member_name.suffix.lower() != ".jsonl"
                or member.get("is_jsonl") is not True
            ):
                raise SelectionError(f"unsafe or non-JSONL member: {member.get('member_path')}")
            if member_name.as_posix() in member_names:
                raise SelectionError(f"duplicate member path in archive: {member_name.as_posix()}")
            member_names.add(member_name.as_posix())
            source_relative = PurePosixPath(str(member.get("extracted_path")))
            if source_relative.is_absolute() or ".." in source_relative.parts:
                raise SelectionError(f"unsafe extracted path: {source_relative.as_posix()}")
            source_path = manifest_base.joinpath(*source_relative.parts).resolve()
            if not source_path.is_relative_to(manifest_base):
                raise SelectionError(f"source path escapes manifest directory: {source_relative}")
            if source_path in source_paths:
                raise SelectionError(f"duplicate resolved source path: {source_relative.as_posix()}")
            source_paths.add(source_path)
            output_path = output_dir / Path(str(archive_name)).stem / Path(*member_name.parts)
            if output_path.resolve() in selected_paths:
                raise SelectionError(f"duplicate selected output path: {member_name.as_posix()}")
            selected_paths.add(output_path.resolve())
            members.append(
                {
                    "archive_name": archive_name,
                    "file_type": file_type,
                    "member_name": member_name.as_posix(),
                    "source_path": source_path,
                    "source_path_record": repository_relative_path(source_path, repository_root),
                    "source_size_bytes": member.get("size_bytes"),
                    "source_sha256": member.get("sha256"),
                    "documented_rows": documented_rows[file_type],
                    "selected_output_path": output_path,
                    "selected_output_path_record": repository_relative_path(
                        output_path, repository_root
                    ),
                }
            )
    if archive_names != set(PE_TEST_ARCHIVES):
        raise SelectionError(f"archive set mismatch: found {sorted(archive_names)}")
    return document, members


def plan_selection(
    manifest_path: Path,
    output_dir: Path,
    repository_root: Path,
    documented_rows: dict[str, int] = DOCUMENTED_ROWS_PER_MEMBER,
    expected_counts: dict[str, int] = EXPECTED_SELECTED_COUNTS,
) -> dict[str, object]:
    _, members = read_source_manifest(
        manifest_path, output_dir, repository_root, documented_rows=documented_rows
    )
    counts = Counter()
    for member in members:
        counts[member["file_type"]] += member["documented_rows"]
    if dict(counts) != expected_counts:
        raise SelectionError(
            f"planned file-type counts mismatch: expected {expected_counts}, found {dict(counts)}"
        )
    return {
        "selection_rule": SELECTION_RULE,
        "source_manifest": repository_relative_path(manifest_path, repository_root),
        "selection_manifest": repository_relative_path(
            output_dir / "selection_manifest.json", repository_root
        ),
        "members": [
            {
                "source": member["source_path_record"],
                "output": member["selected_output_path_record"],
                "file_type": member["file_type"],
                "selected_rows": member["documented_rows"],
            }
            for member in members
        ],
        "selected_counts_by_file_type": dict(counts),
        "total_selected_rows": sum(counts.values()),
    }


def verify_source_members(members: list[dict[str, object]]) -> None:
    for member in members:
        source_path = member["source_path"]
        if not source_path.is_file():
            raise SelectionError(f"missing source member: {member['source_path_record']}")
        observed_size = source_path.stat().st_size
        if observed_size != member["source_size_bytes"]:
            raise SelectionError(f"source size mismatch: {member['source_path_record']}")
        observed_hash = sha256_file(source_path)
        if observed_hash != member["source_sha256"]:
            raise SelectionError(f"source SHA-256 mismatch: {member['source_path_record']}")
        line_count = 0
        with source_path.open("rb") as handle:
            for _ in handle:
                line_count += 1
        expected_rows = 2 * member["documented_rows"]
        if line_count != expected_rows:
            raise SelectionError(
                f"source row count mismatch: {member['source_path_record']} "
                f"expected {expected_rows}, found {line_count}"
            )
        member["source_row_count"] = line_count


def _new_hash_state(
    label: str,
    family: str,
    file_type: str,
    detector_digest: str,
    member_path: str,
    week: str,
) -> list[object]:
    return [1, label, family, file_type, detector_digest, member_path, week, False, False, False, False, False, False]


def update_hash_state(
    states: dict[str, list[object]],
    sample_hash: str,
    record: dict[str, object],
    detector_digest: str,
    member_path: str,
) -> None:
    label = canonical_token(record["label"])
    family = canonical_token(record["family"])
    file_type = canonical_token(record["file_type"])
    week = canonical_token(record["week_id"])
    state = states.get(sample_hash)
    if state is None:
        states[sample_hash] = _new_hash_state(
            label, family, file_type, detector_digest, member_path, week
        )
        return
    state[0] += 1
    state[7] = state[7] or label != state[1]
    state[8] = state[8] or family != state[2]
    state[9] = state[9] or file_type != state[3]
    state[10] = state[10] or detector_digest != state[4]
    state[11] = state[11] or member_path != state[5]
    state[12] = state[12] or week != state[6]


def build_repeat_summary(states: dict[str, list[object]]) -> dict[str, object]:
    multiplicities = Counter(int(state[0]) for state in states.values())
    repeated_hashes = sorted(key for key, state in states.items() if state[0] > 1)
    repeated_digest = hashlib.sha256("\n".join(repeated_hashes).encode("utf-8")).hexdigest()
    return {
        "unique_hash_count": len(states),
        "repeated_hash_count": len(repeated_hashes),
        "repeated_hash_occurrences": sum(int(states[key][0]) for key in repeated_hashes),
        "multiplicity_summary": dict(sorted(multiplicities.items())),
        "repeated_hash_list_sha256": repeated_digest,
        "conflict_counts": {
            "label": sum(1 for key in repeated_hashes if states[key][7]),
            "family": sum(1 for key in repeated_hashes if states[key][8]),
            "file_type": sum(1 for key in repeated_hashes if states[key][9]),
            "detector_input": sum(1 for key in repeated_hashes if states[key][10]),
        },
        "cross_member_and_week_repeated_hash_count": sum(
            1 for key in repeated_hashes if states[key][11] and states[key][12]
        ),
    }


def normalize_repeat_profile(profile: dict[str, object]) -> dict[str, object]:
    if not isinstance(profile, dict) or not isinstance(profile.get("multiplicity_summary"), dict):
        raise SelectionError("residual repeat profile is incomplete")
    normalized = dict(profile)
    normalized["multiplicity_summary"] = {
        int(key): value for key, value in profile["multiplicity_summary"].items()
    }
    return normalized


def validate_repeat_profile(
    observed: dict[str, object], expected: dict[str, object] = REVIEWED_REPEAT_PROFILE
) -> None:
    normalized_expected = normalize_repeat_profile(expected)
    if observed != normalized_expected:
        for key in normalized_expected:
            if observed.get(key) != normalized_expected[key]:
                raise SelectionError(
                    f"reviewed residual repeat profile mismatch for {key}: "
                    f"expected {normalized_expected[key]}, found {observed.get(key)}"
                )
        raise SelectionError("reviewed residual repeat profile mismatch")


def compare_and_write_member(
    member: dict[str, object],
    temporary_path: Path,
    detector_fields: list[str],
    hash_states: dict[str, list[object]],
) -> dict[str, object]:
    required_fields = REQUIRED_PAIR_FIELDS | set(detector_fields)
    selected_rows = member["documented_rows"]
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        member["source_path"].open("rb") as first_handle,
        member["source_path"].open("rb") as second_handle,
        temporary_path.open("wb") as output_handle,
    ):
        for _ in range(selected_rows):
            if not second_handle.readline():
                raise SelectionError(f"unexpected source end: {member['source_path_record']}")
        for offset in range(selected_rows):
            first_line = first_handle.readline()
            second_line = second_handle.readline()
            if not first_line or not second_line:
                raise SelectionError(f"unexpected source end: {member['source_path_record']}")
            try:
                first = json.loads(first_line)
                second = json.loads(second_line)
            except json.JSONDecodeError as error:
                raise SelectionError(
                    f"invalid JSON in {member['source_path_record']} at paired position {offset + 1}"
                ) from error
            if not isinstance(first, dict) or not isinstance(second, dict):
                raise SelectionError(
                    f"non-object JSON in {member['source_path_record']} at paired position {offset + 1}"
                )
            for half_name, record in (("first", first), ("second", second)):
                missing = sorted(required_fields - set(record))
                if missing:
                    raise SelectionError(
                        f"missing required fields in {half_name} half of "
                        f"{member['source_path_record']} at position {offset + 1}: {missing}"
                    )
            if first["sha256"] is None or not str(first["sha256"]).strip():
                raise SelectionError(
                    f"missing or blank SHA-256 in {member['source_path_record']} at position {offset + 1}"
                )
            for field in ("sha256", "label", "family", "file_type", "week_id"):
                if first[field] != second[field]:
                    raise SelectionError(
                        f"paired {field} mismatch in {member['source_path_record']} "
                        f"at position {offset + 1}"
                    )
            if first["file_type"] != member["file_type"]:
                raise SelectionError(
                    f"record file type does not match assigned type in {member['source_path_record']} "
                    f"at position {offset + 1}"
                )
            if (
                not isinstance(first["label"], int)
                or isinstance(first["label"], bool)
                or first["label"] not in (0, 1)
            ):
                raise SelectionError(
                    f"invalid binary label in {member['source_path_record']} "
                    f"at position {offset + 1}"
                )
            differing_detector_fields = [
                field for field in detector_fields if first[field] != second[field]
            ]
            if differing_detector_fields:
                raise SelectionError(
                    f"paired detector-input mismatch in {member['source_path_record']} "
                    f"at position {offset + 1}: {differing_detector_fields}"
                )
            first_detector_digest = detector_projection_digest(first, detector_fields)
            second_detector_digest = detector_projection_digest(second, detector_fields)
            if first_detector_digest != second_detector_digest:
                raise SelectionError(
                    f"paired detector-input mismatch in {member['source_path_record']} "
                    f"at position {offset + 1}"
                )
            differing_fields = {
                field
                for field in set(first) | set(second)
                if field not in first or field not in second or first[field] != second[field]
            }
            unexpected = sorted(differing_fields - set(ALLOWED_METADATA_DIFFERENCES))
            if unexpected:
                raise SelectionError(
                    f"unapproved paired field difference in {member['source_path_record']} "
                    f"at position {offset + 1}: {unexpected}"
                )
            output_handle.write(first_line)
            update_hash_state(
                hash_states,
                str(first["sha256"]),
                first,
                first_detector_digest,
                member["source_path_record"],
            )
        if second_handle.readline():
            raise SelectionError(f"unexpected extra source rows: {member['source_path_record']}")
    return {
        "selected_row_count": selected_rows,
        "selected_output_size_bytes": temporary_path.stat().st_size,
        "selected_output_sha256": sha256_file(temporary_path),
    }


def select_records(
    manifest_path: Path,
    output_dir: Path,
    repository_root: Path,
    extractor: object,
    extractor_version: str,
    overwrite: bool = False,
    documented_rows: dict[str, int] = DOCUMENTED_ROWS_PER_MEMBER,
    expected_counts: dict[str, int] = EXPECTED_SELECTED_COUNTS,
    expected_repeat_profile: dict[str, object] = REVIEWED_REPEAT_PROFILE,
    creation_time: str | None = None,
) -> Path:
    if extractor_version != EXPECTED_THREMBER_VERSION:
        raise SelectionError(
            f"thrember version mismatch: expected {EXPECTED_THREMBER_VERSION}, "
            f"found {extractor_version}"
        )
    extractor_record = inspect_extractor_mapping(extractor)
    _, members = read_source_manifest(
        manifest_path, output_dir, repository_root, documented_rows=documented_rows
    )
    manifest_output_path = output_dir / "selection_manifest.json"
    if not overwrite and output_dir.exists():
        raise SelectionError("completed selection already exists; pass --overwrite to replace it")
    verify_source_members(members)

    staging_dir = create_staging_directory(output_dir)
    selected_records = []
    hash_states: dict[str, list[object]] = {}
    selected_counts = Counter()
    try:
        for member in members:
            relative_output = member["selected_output_path"].relative_to(output_dir)
            staged_output_path = staging_dir / relative_output
            output_record = compare_and_write_member(
                member,
                staged_output_path,
                extractor_record["detector_input_fields"],
                hash_states,
            )
            selected_counts[member["file_type"]] += output_record["selected_row_count"]
            selected_records.append(
                {
                    "archive_name": member["archive_name"],
                    "member_name": member["member_name"],
                    "file_type": member["file_type"],
                    "source_member_path": member["source_path_record"],
                    "source_member_size_bytes": member["source_size_bytes"],
                    "source_member_sha256": member["source_sha256"],
                    "source_row_count": member["source_row_count"],
                    "documented_row_count": member["documented_rows"],
                    "selected_output_path": member["selected_output_path_record"],
                    **output_record,
                }
            )
        if dict(selected_counts) != expected_counts:
            raise SelectionError(
                f"selected file-type counts mismatch: expected {expected_counts}, "
                f"found {dict(selected_counts)}"
            )
        total_selected = sum(selected_counts.values())
        if total_selected != sum(expected_counts.values()):
            raise SelectionError("selected total row count mismatch")
        repeat_summary = build_repeat_summary(hash_states)
        validate_repeat_profile(repeat_summary, expected_repeat_profile)
        manifest_document = {
            "schema_version": 1,
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "selection_rule_name": SELECTION_RULE,
            "source_manifest": {
                "path": repository_relative_path(manifest_path, repository_root),
                "sha256": sha256_file(manifest_path),
            },
            "selected_member_order": selected_records,
            "detector_input_fields": extractor_record["detector_input_fields"],
            "extractor_dimension": extractor_record["dimension"],
            "extractor_version": extractor_version,
            "extractor_components": extractor_record["feature_components"],
            "allowed_metadata_only_differences": ALLOWED_METADATA_DIFFERENCES,
            "total_selected_counts_by_file_type": dict(selected_counts),
            "total_selected_row_count": total_selected,
            **repeat_summary,
            "creation_time": creation_time or utc_now(),
            "completion_status": "complete",
            "residual_repeat_policy": (
                "The 60 reviewed cross-week repeated hashes remain in the primary data. "
                "A separate unique-hash sensitivity analysis is planned after inference."
            ),
        }
        staged_manifest_path = staging_dir / "selection_manifest.json"
        staged_manifest_path.write_text(
            json.dumps(manifest_document, indent=2) + "\n", encoding="utf-8"
        )
        validate_staged_publication(staging_dir, selected_records)
        publish_staged_selection(staging_dir, output_dir, overwrite)
        return manifest_output_path
    except Exception:
        remove_operation_directory(staging_dir, output_dir, "staging")
        raise


def summarize_selected_files(
    selected_members: list[dict[str, object]],
    repository_root: Path,
    detector_fields: list[str],
) -> tuple[list[tuple[str, Path]], dict[str, object]]:
    hash_states: dict[str, list[object]] = {}
    inputs = []
    for member in selected_members:
        selected_path = resolve_repository_path(member.get("selected_output_path"), repository_root)
        if not selected_path.is_file():
            raise SelectionError(f"missing selected output: {member.get('selected_output_path')}")
        if selected_path.stat().st_size != member.get("selected_output_size_bytes"):
            raise SelectionError(
                f"selected output size mismatch: {member.get('selected_output_path')}"
            )
        if sha256_file(selected_path) != member.get("selected_output_sha256"):
            raise SelectionError(
                f"selected output SHA-256 mismatch: {member.get('selected_output_path')}"
            )
        row_count = 0
        with selected_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise SelectionError(
                        f"invalid JSON in selected output {member.get('selected_output_path')} "
                        f"at line {line_number}"
                    ) from error
                required = REQUIRED_PAIR_FIELDS | set(detector_fields)
                missing = sorted(required - set(record))
                if missing:
                    raise SelectionError(
                        f"missing required fields in selected output "
                        f"{member.get('selected_output_path')}: {missing}"
                    )
                sample_hash = record["sha256"]
                if sample_hash is None or not str(sample_hash).strip():
                    raise SelectionError(
                        f"missing or blank SHA-256 in selected output: "
                        f"{member.get('selected_output_path')}"
                    )
                if record["file_type"] != member.get("file_type"):
                    raise SelectionError(
                        f"selected record file type mismatch: {member.get('selected_output_path')}"
                    )
                detector_digest = detector_projection_digest(record, detector_fields)
                update_hash_state(
                    hash_states,
                    str(sample_hash),
                    record,
                    detector_digest,
                    str(member.get("source_member_path")),
                )
                row_count += 1
        if row_count != member.get("selected_row_count"):
            raise SelectionError(
                f"selected output row count mismatch: {member.get('selected_output_path')}"
            )
        inputs.append((str(member.get("file_type")), selected_path))
    return inputs, build_repeat_summary(hash_states)


def load_completed_selection_manifest(
    manifest_path: Path,
    repository_root: Path,
    extractor: object,
    expected_counts: dict[str, int] = EXPECTED_SELECTED_COUNTS,
    expected_repeat_profile: dict[str, object] = REVIEWED_REPEAT_PROFILE,
    documented_rows: dict[str, int] = DOCUMENTED_ROWS_PER_MEMBER,
) -> tuple[dict[str, object], list[tuple[str, Path]]]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise SelectionError("selection manifest has an unsupported schema version")
    if document.get("dataset_repository") != DATASET_REPOSITORY:
        raise SelectionError("selection manifest has the wrong dataset repository")
    if document.get("dataset_revision") != DATASET_REVISION:
        raise SelectionError("selection manifest has the wrong dataset revision")
    if document.get("selection_rule_name") != SELECTION_RULE:
        raise SelectionError("selection manifest has the wrong selection rule")
    if document.get("completion_status") != "complete":
        raise SelectionError("selection manifest is incomplete")
    extractor_record = inspect_extractor_mapping(extractor)
    if document.get("detector_input_fields") != extractor_record["detector_input_fields"]:
        raise SelectionError("selection manifest detector-input fields do not match the extractor")
    if document.get("extractor_dimension") != extractor_record["dimension"]:
        raise SelectionError("selection manifest extractor dimension does not match the extractor")
    if document.get("extractor_version") != EXPECTED_THREMBER_VERSION:
        raise SelectionError("selection manifest has the wrong thrember version")
    if document.get("allowed_metadata_only_differences") != ALLOWED_METADATA_DIFFERENCES:
        raise SelectionError("selection manifest has the wrong allowed metadata differences")
    if document.get("total_selected_counts_by_file_type") != expected_counts:
        raise SelectionError("selection manifest has the wrong file-type totals")
    if document.get("total_selected_row_count") != sum(expected_counts.values()):
        raise SelectionError("selection manifest has the wrong total row count")
    source_manifest = document.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise SelectionError("selection manifest has no source manifest record")
    source_path = resolve_repository_path(source_manifest.get("path"), repository_root)
    if not source_path.is_file() or sha256_file(source_path) != source_manifest.get("sha256"):
        raise SelectionError("source download manifest does not match the selection manifest")
    selected_members = document.get("selected_member_order")
    if not isinstance(selected_members, list) or not selected_members:
        raise SelectionError("selection manifest has no selected members")
    _, expected_members = read_source_manifest(
        source_path,
        manifest_path.parent,
        repository_root,
        documented_rows=documented_rows,
    )
    if len(selected_members) != len(expected_members):
        raise SelectionError("selection manifest member count does not match the source manifest")
    selected_counts = Counter()
    selected_output_paths = set()
    for selected, expected in zip(selected_members, expected_members, strict=True):
        expected_record = {
            "archive_name": expected["archive_name"],
            "member_name": expected["member_name"],
            "file_type": expected["file_type"],
            "source_member_path": expected["source_path_record"],
            "source_member_size_bytes": expected["source_size_bytes"],
            "source_member_sha256": expected["source_sha256"],
            "source_row_count": 2 * expected["documented_rows"],
            "documented_row_count": expected["documented_rows"],
            "selected_output_path": expected["selected_output_path_record"],
            "selected_row_count": expected["documented_rows"],
        }
        for key, value in expected_record.items():
            if selected.get(key) != value:
                raise SelectionError(
                    f"selection member order or {key} does not match the source manifest"
                )
        selected_counts[selected["file_type"]] += selected["selected_row_count"]
        output_path = selected["selected_output_path"]
        if output_path in selected_output_paths:
            raise SelectionError("selection manifest repeats a selected output path")
        selected_output_paths.add(output_path)
    if dict(selected_counts) != expected_counts:
        raise SelectionError("selected member file-type counts do not match reviewed totals")
    recorded_rows = sum(int(member.get("selected_row_count", -1)) for member in selected_members)
    if recorded_rows != document["total_selected_row_count"]:
        raise SelectionError("selected member rows do not match the selection total")
    inputs, observed_profile = summarize_selected_files(
        selected_members, repository_root, extractor_record["detector_input_fields"]
    )
    validate_repeat_profile(observed_profile, expected_repeat_profile)
    recorded_profile = {
        key: document.get(key)
        for key in (
            "unique_hash_count",
            "repeated_hash_count",
            "repeated_hash_occurrences",
            "multiplicity_summary",
            "repeated_hash_list_sha256",
            "conflict_counts",
            "cross_member_and_week_repeated_hash_count",
        )
    }
    validate_repeat_profile(normalize_repeat_profile(recorded_profile), expected_repeat_profile)
    return document, inputs
