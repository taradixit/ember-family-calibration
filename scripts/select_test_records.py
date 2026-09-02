#!/usr/bin/env python3
"""Select the reviewed first half of each verified EMBER2024 PE test member."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember_calibration.selection import plan_selection, select_records  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="write the reviewed selection")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing completed selection; valid only with --execute",
    )
    args = parser.parse_args(argv)
    if args.overwrite and not args.execute:
        parser.error("--overwrite requires --execute")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    if not args.execute:
        plan = plan_selection(
            args.download_manifest,
            args.output_dir,
            repository_root,
        )
        print("dry run: source JSONL files will not be read and no selection will be written")
        print(json.dumps(plan, indent=2))
        return

    from thrember import PEFeatureExtractor

    manifest = select_records(
        args.download_manifest,
        args.output_dir,
        repository_root,
        PEFeatureExtractor(),
        version("thrember"),
        overwrite=args.overwrite,
    )
    print(f"completed selection manifest: {manifest.relative_to(repository_root).as_posix()}")


if __name__ == "__main__":
    main()
