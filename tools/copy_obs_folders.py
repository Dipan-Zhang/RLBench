#!/usr/bin/env python3
"""Copy task obs folders from one output tree to another.

Default behavior:
  outputs_ablation_2D/<task>/obs -> outputs_autopipe/<task>/obs

If outputs_autopipe/<task> already exists, the task is skipped.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy <task>/obs folders into another output directory."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "outputs_ablation_2D",
        help="Source output directory containing <task>/obs folders.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "outputs_autopipe",
        help="Destination output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without creating files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")

    copied = 0
    skipped_existing = 0
    skipped_missing_obs = 0

    for task_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        source_obs = task_dir / "obs"
        dest_task = dest / task_dir.name
        dest_obs = dest_task / "obs"

        if not source_obs.is_dir():
            skipped_missing_obs += 1
            print(f"missing obs, skip: {task_dir.name}")
            continue

        if dest_task.exists():
            skipped_existing += 1
            print(f"existing task, skip: {task_dir.name}")
            continue

        print(f"copy: {source_obs} -> {dest_obs}")
        if not args.dry_run:
            dest_task.mkdir(parents=True, exist_ok=False)
            shutil.copytree(source_obs, dest_obs)
        copied += 1

    action = "would copy" if args.dry_run else "copied"
    print(
        f"{action}: {copied}; "
        f"skipped existing tasks: {skipped_existing}; "
        f"skipped missing obs: {skipped_missing_obs}"
    )


if __name__ == "__main__":
    main()
