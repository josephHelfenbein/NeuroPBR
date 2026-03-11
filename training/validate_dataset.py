#!/usr/bin/env python3
"""
Dataset validation for NeuroPBR.

Scans image files and/or shards for corruption, truncation, and NaN/Inf values.
By default runs in dry-run mode (report only). Use --delete to remove corrupted files.

Usage:
    python validate_dataset.py --input-dir /path/to/input --output-dir /path/to/output
    python validate_dataset.py --input-dir /path/to/input --output-dir /path/to/output --delete
    python validate_dataset.py --shards-dir /path/to/shards
    python validate_dataset.py --input-dir /path/to/input --output-dir /path/to/output --shards-dir /path/to/shards
"""

import argparse
import sys
from pathlib import Path
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch
import numpy as np

# Allow loading truncated images for detection
ImageFile.LOAD_TRUNCATED_IMAGES = True


def validate_image(path: Path) -> tuple[Path, bool, str]:
    """Validate a single image file. Returns (path, is_valid, error_message)."""
    try:
        with Image.open(path) as img:
            img.verify()

        # Re-open after verify (verify closes the file)
        with Image.open(path) as img:
            img.load()
            if img.size[0] < 1 or img.size[1] < 1:
                return (path, False, f"Invalid dimensions: {img.size}")
            img_data = np.array(img)
            if np.isnan(img_data).any():
                return (path, False, "Contains NaN values")

        return (path, True, "")
    except Exception as e:
        return (path, False, str(e))


def find_all_images(directory: Path, extensions: set = {'.png', '.jpg', '.jpeg', '.exr'}) -> list[Path]:
    """Recursively find all image files in a directory."""
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))
    return images


def validate_images(dirs: list[Path], workers: int = 8) -> list[tuple[Path, str]]:
    """Validate all images across the given directories. Returns list of (path, error)."""
    all_images = []
    for d in dirs:
        if d.exists():
            all_images.extend(find_all_images(d))
        else:
            print(f"Warning: directory does not exist: {d}")

    if not all_images:
        print("No images found.")
        return []

    print(f"Found {len(all_images)} image files")

    corrupted = []
    valid_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(validate_image, path): path for path in all_images}
        with tqdm(total=len(all_images), desc="Validating images") as pbar:
            for future in as_completed(futures):
                path, is_valid, error = future.result()
                if is_valid:
                    valid_count += 1
                else:
                    corrupted.append((path, error))
                pbar.update(1)

    print(f"  Valid: {valid_count}  Corrupted: {len(corrupted)}")
    return corrupted


def validate_shards(shards_dir: Path) -> list[tuple[Path, str]]:
    """Check shards for NaN/Inf values. Returns list of (path, error)."""
    shards = sorted(shards_dir.glob("shard_*.pt"))
    if not shards:
        print(f"No shards found in {shards_dir}")
        return []

    print(f"Found {len(shards)} shards")
    corrupted = []

    for shard_path in tqdm(shards, desc="Checking shards"):
        try:
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
            teacher_outputs = data.get("teacher_outputs", {})
            for k, v in teacher_outputs.items():
                if isinstance(v, torch.Tensor) and (torch.isnan(v).any() or torch.isinf(v).any()):
                    corrupted.append((shard_path, f"NaN/Inf in key '{k}'"))
                    break
        except Exception as e:
            corrupted.append((shard_path, f"Failed to load: {e}"))

    valid = len(shards) - len(corrupted)
    print(f"  Valid: {valid}  Corrupted: {len(corrupted)}")
    return corrupted


def main():
    parser = argparse.ArgumentParser(
        description="Validate NeuroPBR dataset images and shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validate_dataset.py --input-dir ./data/input --output-dir ./data/output
    python validate_dataset.py --input-dir ./data/input --output-dir ./data/output --delete
    python validate_dataset.py --shards-dir ./shards
        """
    )
    parser.add_argument("--input-dir", type=str, help="Input renders directory")
    parser.add_argument("--output-dir", type=str, help="Output PBR maps directory")
    parser.add_argument("--shards-dir", type=str, help="Shards directory to check for NaN/Inf")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for image validation (default: 8)")
    parser.add_argument("--delete", action="store_true", help="Delete corrupted files (default: dry-run)")
    args = parser.parse_args()

    if not args.input_dir and not args.shards_dir:
        parser.error("At least one of --input-dir or --shards-dir is required")

    all_corrupted: list[tuple[Path, str]] = []

    # Validate images
    if args.input_dir:
        image_dirs = [Path(args.input_dir)]
        if args.output_dir:
            image_dirs.append(Path(args.output_dir))
        print("Scanning images...")
        all_corrupted.extend(validate_images(image_dirs, workers=args.workers))

    # Validate shards
    if args.shards_dir:
        shards_path = Path(args.shards_dir)
        if not shards_path.exists():
            print(f"Error: shards directory does not exist: {shards_path}")
            sys.exit(1)
        print("\nScanning shards...")
        all_corrupted.extend(validate_shards(shards_path))

    # Report
    print(f"\n{'='*60}")
    if not all_corrupted:
        print("All files are valid.")
        sys.exit(0)

    print(f"Found {len(all_corrupted)} corrupted file(s):")
    print(f"{'='*60}")
    for path, error in all_corrupted:
        print(f"  {path}")
        print(f"    {error}")

    if args.delete:
        print(f"\nDeleting {len(all_corrupted)} corrupted file(s)...")
        deleted = 0
        for path, _ in all_corrupted:
            try:
                path.unlink()
                deleted += 1
            except Exception as e:
                print(f"  Failed to remove {path}: {e}")
        print(f"Deleted {deleted}/{len(all_corrupted)} files.")
    else:
        print(f"\nDry run — no files deleted. Re-run with --delete to remove them.")

    sys.exit(1 if all_corrupted else 0)


if __name__ == "__main__":
    main()
