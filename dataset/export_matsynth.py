from datasets import load_dataset
from PIL import Image
from pathlib import Path
import json


import argparse

def save_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    (img if isinstance(img, Image.Image) else Image.fromarray(img)).save(path)


def main(dst="matsynth_raw", split="train", limit=100, save_metadata=True, start=0):
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)

    if start > limit:
        raise ValueError("start must be less than limit")

    if start > 0:
        print(f"Starting from index {start}")

    ds = load_dataset("gvecchio/MatSynth", streaming=True, split=split)
    i = 0
    # keys we will export as images, preserving original names
    image_keys = [
        "basecolor", "normal", "roughness", "metallic",
        "diffuse", "specular", "displacement", "opacity", "blend_mask",
    ]
    for ex in ds:
        # ensure at least one of the core channels exists
        if not any(ex.get(k) is not None for k in ("basecolor", "normal", "roughness", "metallic")):
            continue
        
        if i < start:
            i += 1
            continue

        mdir = out / f"mat_{i:05d}"
        for key in image_keys:
            if ex.get(key) is not None:
                save_image(ex[key], mdir / f"{key}.png")
        if save_metadata and ex.get("metadata") is not None:
            mdir.mkdir(parents=True, exist_ok=True)
            with open(mdir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(ex["metadata"], f, indent=2)
        i += 1
        if i >= limit:
            break
    print(f"Exported {i} materials to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MatSynth dataset to local disk.")
    parser.add_argument("--dst", type=str, default="matsynth_raw", help="Destination directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of materials to export")
    parser.add_argument("--no-metadata", action="store_false", dest="save_metadata", help="Do not save metadata.json")
    parser.add_argument("--start", type=int, default=0, help="Start index to resume export")
    
    args = parser.parse_args()
    
    main(
        dst=args.dst,
        split=args.split,
        limit=args.limit,
        save_metadata=args.save_metadata,
        start=args.start
    )
