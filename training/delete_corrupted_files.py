#!/usr/bin/env python3
"""
Delete corrupted image files listed in a corrupted_files.json produced by validate_dataset.py.

The script removes ONLY files; it will not touch directories or render_metadata.json.

Usage:
    python delete_corrupted_files.py --corrupted_json corrupted_files.json [--dry-run]
"""

import argparse
import json
from pathlib import Path


def load_corrupted_list(path: Path):
    data = json.loads(path.read_text())
    # Support both list of dicts and simple list of paths
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "path" in data[0]:
            return [Path(item["path"]) for item in data]
        return [Path(p) for p in data]
    raise ValueError("Unsupported corrupted_files.json format; expected list")


def main():
    parser = argparse.ArgumentParser(description="Delete corrupted image files listed in a JSON report")
    parser.add_argument("--corrupted_json", type=str, default="corrupted_files.json", help="Path to corrupted_files.json")
    parser.add_argument("--dry-run", action="store_true", help="List files but do not delete")
    args = parser.parse_args()

    json_path = Path(args.corrupted_json)
    if not json_path.exists():
        raise FileNotFoundError(f"corrupted json not found: {json_path}")

    files = load_corrupted_list(json_path)
    total = len(files)
    deleted = 0
    skipped_missing = 0
    skipped_not_file = 0
    dry_counts = {}

    for p in files:
        if not p.exists():
            skipped_missing += 1
            continue
        if not p.is_file():
            skipped_not_file += 1
            continue
        if args.dry_run:
            print(f"[dry-run] would delete: {p}")
            dry_counts[p.parent] = dry_counts.get(p.parent, 0) + 1
        else:
            p.unlink()
            deleted += 1

    print("Deletion summary:")
    print(f"  total listed:     {total}")
    print(f"  deleted:          {deleted}")
    print(f"  missing skipped:  {skipped_missing}")
    print(f"  not-file skipped: {skipped_not_file}")
    if args.dry_run:
        print("Dry run only; no files were removed.")
        if dry_counts:
            print("Per-directory counts:")
            for parent, count in sorted(dry_counts.items()):
                print(f"  {parent}: {count}")


if __name__ == "__main__":
    main()
