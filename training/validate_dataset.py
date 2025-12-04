#!/usr/bin/env python3
"""
Dataset Validation Script for NeuroPBR

Scans all image files in the dataset to identify corrupted or truncated files.
Run this before training to catch problematic files early.

Usage:
    python validate_dataset.py --input_dir /path/to/input --output_dir /path/to/output
"""

import argparse
from pathlib import Path
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

# Allow loading truncated images for detection
ImageFile.LOAD_TRUNCATED_IMAGES = True


def validate_image(path: Path) -> tuple[Path, bool, str]:
    """
    Validate a single image file.
    
    Returns:
        (path, is_valid, error_message)
    """
    try:
        with Image.open(path) as img:
            img.load()  # Force full load to detect truncation
            img.verify()  # Additional verification
        
        # Re-open after verify (verify closes the file)
        with Image.open(path) as img:
            img.load()
            # Check if image has reasonable dimensions
            if img.size[0] < 1 or img.size[1] < 1:
                return (path, False, f"Invalid dimensions: {img.size}")
            
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


def main():
    parser = argparse.ArgumentParser(description="Validate NeuroPBR dataset images")
    parser.add_argument("--input_dir", type=str, required=True, help="Input renders directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output PBR maps directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--fix", action="store_true", help="Attempt to remove corrupted files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find all images
    print("Scanning for image files...")
    all_images = []
    
    if input_dir.exists():
        all_images.extend(find_all_images(input_dir))
    else:
        print(f"Warning: Input directory does not exist: {input_dir}")
        
    if output_dir.exists():
        all_images.extend(find_all_images(output_dir))
    else:
        print(f"Warning: Output directory does not exist: {output_dir}")

    print(f"Found {len(all_images)} image files to validate")

    if not all_images:
        print("No images found. Check your paths.")
        return

    # Validate images in parallel
    corrupted = []
    valid_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(validate_image, path): path for path in all_images}
        
        with tqdm(total=len(all_images), desc="Validating images") as pbar:
            for future in as_completed(futures):
                path, is_valid, error = future.result()
                if is_valid:
                    valid_count += 1
                else:
                    corrupted.append((path, error))
                pbar.update(1)

    # Report results
    print(f"\n{'='*60}")
    print(f"Validation Complete")
    print(f"{'='*60}")
    print(f"Total images: {len(all_images)}")
    print(f"Valid: {valid_count}")
    print(f"Corrupted: {len(corrupted)}")

    if corrupted:
        print(f"\n{'='*60}")
        print("Corrupted files:")
        print(f"{'='*60}")
        for path, error in corrupted:
            print(f"  {path}")
            print(f"    Error: {error}")
        
        # Save list of corrupted files
        corrupted_list_path = Path("corrupted_files.json")
        with open(corrupted_list_path, 'w') as f:
            json.dump([{"path": str(p), "error": e} for p, e in corrupted], f, indent=2)
        print(f"\nCorrupted file list saved to: {corrupted_list_path}")

        if args.fix:
            print("\nRemoving corrupted files...")
            for path, _ in corrupted:
                try:
                    path.unlink()
                    print(f"  Removed: {path}")
                except Exception as e:
                    print(f"  Failed to remove {path}: {e}")
    else:
        print("\n✓ All images are valid!")


if __name__ == "__main__":
    main()
