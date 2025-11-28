import argparse
import io
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from datasets import load_dataset
from PIL import Image

# Try to import google cloud storage, but don't fail if not present unless needed
try:
    from google.cloud import storage
except ImportError:
    storage = None

# Clean Dataset Helpers (from clean_dataset.py)
TARGET_MAPS: Tuple[str, ...] = ("albedo", "metallic", "roughness", "normal")
ALLOWED_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".webp"}

def debug(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^a-z0-9\-]+", "", text)
    text = re.sub(r"\-+", "-", text)
    return text or "material"

_MAP_PATTERNS: Dict[str, List[re.Pattern]] = {
    "albedo": [
        re.compile(r"\balbedo\b", re.IGNORECASE),
        re.compile(r"\bbase\s*color\b|\bbasecolor\b|\bbc\b", re.IGNORECASE),
        re.compile(r"\bbase\s*colour\b|\bbasecolour\b", re.IGNORECASE),
        re.compile(r"\bdiffuse\b|\bdiff\b", re.IGNORECASE),
    ],
    "metallic": [
        re.compile(r"\bmetallic\b|\bmetalness\b|\bmetalic\b", re.IGNORECASE),
        re.compile(r"\bmetal\b|\bmet\b|\bmtl\b", re.IGNORECASE),
    ],
    "roughness": [
        re.compile(r"\broughness\b|\brough\b|\brgh\b", re.IGNORECASE),
    ],
    "normal": [
        re.compile(r"\bnormal\b|\bnormals\b|\bnrm\b|\bnor\b", re.IGNORECASE),
        re.compile(r"(?:^|[_\-\.])n(?:[_\-\.]|$)", re.IGNORECASE),
        re.compile(r"\bnormalgl\b|\bnorm\s*gl\b", re.IGNORECASE),
    ],
}

def strip_udim_token(name_no_ext: str) -> str:
    return re.sub(r"([_\.])1\d{3}$", "", name_no_ext)

def guess_map_type(filename: str) -> Optional[str]:
    name = Path(filename).name
    name_no_ext = os.path.splitext(name)[0]
    name_no_udim = strip_udim_token(name_no_ext)
    normalized = re.sub(r"[\._\-]+", " ", name_no_udim).lower()
    for map_type in ("normal", "roughness", "metallic", "albedo"):
        for pat in _MAP_PATTERNS[map_type]:
            if pat.search(normalized):
                return map_type
    return None

@dataclass
class MaterialEntry:
    material_root: Path
    material_slug: str
    files_by_map: Dict[str, Path] = field(default_factory=dict)

def find_materials(src_dir: Path, allowed_exts: Set[str], verbose: bool) -> List[MaterialEntry]:
    materials: Dict[Path, MaterialEntry] = {}
    for root, _dirs, files in os.walk(src_dir):
        root_path = Path(root)
        if any(p.startswith(".") for p in root_path.parts):
            continue
        files_in_dir = [f for f in files if not f.startswith(".")]
        for fname in files_in_dir:
            ext = Path(fname).suffix.lower()
            if ext not in allowed_exts:
                continue
            map_type = guess_map_type(fname)
            if map_type is None:
                continue
            entry = materials.get(root_path)
            if entry is None:
                material_slug = slugify(root_path.name)
                entry = MaterialEntry(material_root=root_path, material_slug=material_slug)
                materials[root_path] = entry
            existing = entry.files_by_map.get(map_type)
            candidate = root_path / fname
            if existing is None or len(candidate.name) < len(existing.name):
                entry.files_by_map[map_type] = candidate
                debug(f"Detected {map_type}: {candidate}", verbose)
    return list(materials.values())

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path, link: bool, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            if dst.is_file() or dst.is_symlink():
                dst.unlink()
        else:
            return
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)

def convert_to_png(src: Path, dst: Path, map_type: str, verbose: bool, resize: Optional[int] = None) -> bool:
    try:
        with Image.open(src) as im:
            if resize is not None and (im.width != resize or im.height != resize):
                resample = getattr(Image, "Resampling", Image).LANCZOS
                im = im.resize((resize, resize), resample)
            if map_type in ("albedo", "normal"):
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                elif im.mode == "RGBA":
                    im = im.convert("RGB")
            else:
                if im.mode not in ("L", "I;16"):
                    im = im.convert("L")
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="PNG", optimize=True)
        return True
    except Exception as e:
        print(f"ERROR: PNG conversion failed for {src}: {e}")
        return False

def clean_dataset_logic(src_dir: Path, dst_dir: Path, flatten: bool, require_all: bool, 
                        link: bool, overwrite: bool, dry_run: bool, verbose: bool, 
                        keep_ext: bool, allowed_exts: Set[str], manifest_path: Optional[Path], 
                        resize: Optional[int]):
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"ERROR: --src does not exist or is not a directory: {src_dir}")
        return

    ensure_dir(dst_dir)
    debug(f"Scanning source: {src_dir}", verbose)
    materials = find_materials(src_dir, allowed_exts, verbose)
    included = []
    total_candidates = 0
    total_included = 0

    for entry in sorted(materials, key=lambda m: m.material_slug):
        total_candidates += 1
        present_maps = set(entry.files_by_map.keys())
        if require_all and not all(m in present_maps for m in TARGET_MAPS):
            continue
        maps_to_use = [m for m in TARGET_MAPS if m in present_maps]
        if not maps_to_use:
            continue

        if flatten:
            dest_dir = dst_dir
        else:
            dest_dir = dst_dir / entry.material_slug
        if not dry_run:
            ensure_dir(dest_dir)

        record = {"material": entry.material_slug}
        for map_type in maps_to_use:
            src_path = entry.files_by_map[map_type]
            ext = src_path.suffix
            if flatten:
                dst_path = dest_dir / (f"{entry.material_slug}_{map_type}.png" if not keep_ext else f"{entry.material_slug}_{map_type}{ext}")
            else:
                dst_path = dest_dir / (f"{map_type}.png" if not keep_ext else f"{map_type}{ext}")

            if dry_run:
                print(f"Would process {src_path} -> {dst_path}")
            else:
                if keep_ext:
                    copy_or_link(src_path, dst_path, link=link, overwrite=overwrite)
                else:
                    success = convert_to_png(src_path, dst_path, map_type, verbose, resize=resize)
                    if not success:
                        fallback_dst = dst_path.with_suffix(ext)
                        copy_or_link(src_path, fallback_dst, link=link, overwrite=overwrite)
                        dst_path = fallback_dst
            record[map_type] = str(dst_path)
        included.append(record)
        total_included += 1

    if manifest_path is not None and not dry_run:
        ensure_dir(manifest_path.parent)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"materials": included}, f, indent=2)
    
    print(f"Cleaning finished. Scanned: {total_candidates}, included: {total_included} (dst: {dst_dir})")

# --- Export Logic ---

def save_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    (img if isinstance(img, Image.Image) else Image.fromarray(img)).save(path)

def export_local(dst="matsynth_raw", split="train", limit=100, save_metadata=True, start=0):
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    if start > limit:
        raise ValueError("start must be less than limit")
    if start > 0:
        print(f"Starting export from index {start}")

    ds = load_dataset("gvecchio/MatSynth", streaming=True, split=split)
    i = 0
    image_keys = ["basecolor", "normal", "roughness", "metallic", "diffuse", "specular", "displacement", "opacity", "blend_mask"]
    
    for ex in ds:
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
        if i % 10 == 0:
            print(f"Exported {i}/{limit} materials locally")
        if i >= limit:
            break
    print(f"Exported {i} materials to {out}")

def export_gcs_direct(bucket_name, prefix="raw", limit=4000, start=0):
    if storage is None:
        print("Error: google-cloud-storage not installed.")
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    if start > limit:
        raise ValueError("start must be less than limit")
    if start > 0:
        print(f"Starting GCS export from index {start}")
    
    ds = load_dataset("gvecchio/MatSynth", streaming=True, split="train")
    i = 0
    image_keys = ["basecolor", "normal", "roughness", "metallic", "diffuse", "specular", "displacement", "opacity", "blend_mask"]
    
    for ex in ds:
        if not any(ex.get(k) for k in ("basecolor", "normal", "roughness", "metallic")):
            continue
        if i < start:
            i += 1
            continue
        
        for key in image_keys:
            if ex.get(key) is not None:
                buf = io.BytesIO()
                img = ex[key] if isinstance(ex[key], Image.Image) else Image.fromarray(ex[key])
                img.save(buf, format='PNG')
                buf.seek(0)
                blob = bucket.blob(f"{prefix}/mat_{i:05d}/{key}.png")
                blob.upload_from_file(buf, content_type='image/png')
        i += 1
        if i % 10 == 0:
            print(f"Uploaded {i}/{limit} materials to GCS")
        if i >= limit:
            break
    print(f"Done! Exported {i} materials to GCS")

def upload_folder_to_gcs(source_folder, bucket_name, prefix):
    if storage is None:
        print("Error: google-cloud-storage not installed.")
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    source_path = Path(source_folder)
    
    print(f"Uploading {source_folder} to gs://{bucket_name}/{prefix}...")
    
    for root, _, files in os.walk(source_path):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(source_path)
            blob_path = f"{prefix}/{relative_path.as_posix()}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))
            # print(f"Uploaded {blob_path}")
    print("Upload complete.")

# Stream and Clean Logic

def process_image_in_memory(img, resize=None, map_type="albedo"):
    # Convert to PIL if needed
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # Resize
    if resize is not None and (img.width != resize or img.height != resize):
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((resize, resize), resample)
    
    # Normalize mode
    if map_type in ("albedo", "normal"):
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            img = img.convert("RGB")
    else: # roughness, metallic
        if img.mode not in ("L", "I;16"):
            img = img.convert("L")
            
    return img

def stream_and_clean(
    split="train",
    limit=100,
    start=0,
    dst_dir=None, # Path object or None
    bucket_name=None, # str or None
    prefix="clean",
    resize=2048,
    flatten=False,
    manifest_path=None # Path object
):
    # Setup GCS if needed
    bucket = None
    if bucket_name:
        if storage is None:
            print("Error: google-cloud-storage not installed.")
            return
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        print(f"Streaming cleaned data to GCS bucket: {bucket_name}")
    elif dst_dir:
        ensure_dir(dst_dir)
        print(f"Streaming cleaned data to local directory: {dst_dir}")
    else:
        print("Error: No destination specified (dst_dir or bucket_name required)")
        return

    if start > limit:
        raise ValueError("start must be less than limit")
    if start > 0:
        print(f"Starting stream from index {start}")

    ds = load_dataset("gvecchio/MatSynth", streaming=True, split=split)
    i = 0
    
    # Mapping from MatSynth keys to our target names
    # MatSynth keys: basecolor, normal, roughness, metallic, diffuse, specular, displacement, opacity, blend_mask
    key_map = {
        "basecolor": "albedo",
        "diffuse": "albedo", # Fallback
        "normal": "normal",
        "roughness": "roughness",
        "metallic": "metallic"
    }
    
    included = []
    
    for ex in ds:
        # Check for required keys (we need at least one valid set of PBR maps)
        # MatSynth usually has basecolor, normal, roughness, metallic
        has_base = ex.get("basecolor") is not None or ex.get("diffuse") is not None
        has_normal = ex.get("normal") is not None
        has_rough = ex.get("roughness") is not None
        has_metal = ex.get("metallic") is not None
        
        if not (has_base and has_normal and has_rough and has_metal):
            continue
            
        if i < start:
            i += 1
            continue
            
        slug = f"mat_{i:05d}"
        record = {"material": slug}
        
        # Process each map type
        for src_key, target_map in key_map.items():
            # Skip diffuse if basecolor is present (prefer basecolor)
            if src_key == "diffuse" and ex.get("basecolor") is not None:
                continue
            
            if ex.get(src_key) is None:
                continue
                
            # Process image
            img = process_image_in_memory(ex[src_key], resize=resize, map_type=target_map)
            
            # Determine filename
            filename = f"{target_map}.png"
            if flatten:
                filename = f"{slug}_{target_map}.png"
            
            # Save
            if bucket:
                # Upload to GCS
                blob_path = f"{prefix}/{slug}/{filename}" if not flatten else f"{prefix}/{filename}"
                buf = io.BytesIO()
                img.save(buf, format='PNG', optimize=True)
                buf.seek(0)
                blob = bucket.blob(blob_path)
                blob.upload_from_file(buf, content_type='image/png')
                record[target_map] = f"gs://{bucket_name}/{blob_path}"
            else:
                # Save locally
                if flatten:
                    out_path = dst_dir / filename
                else:
                    mat_dir = dst_dir / slug
                    ensure_dir(mat_dir)
                    out_path = mat_dir / filename
                
                img.save(out_path, format='PNG', optimize=True)
                record[target_map] = str(out_path)
        
        included.append(record)
        
        i += 1
        if i % 10 == 0:
            print(f"Processed {i}/{limit} materials")
            
        if i >= limit:
            break
            
    # Save manifest if local
    if manifest_path and dst_dir:
        ensure_dir(manifest_path.parent)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"materials": included}, f, indent=2)
            
    print(f"Done! Processed {i} materials.")


def main():
    parser = argparse.ArgumentParser(description="Consolidated MatSynth Dataset Tool")
    
    # Mode selection
    parser.add_argument("--clean", action="store_true", help="Clean the dataset (stream & clean in memory if exporting, or clean from disk)")
    parser.add_argument("--gcs", action="store_true", help="Export/Upload to Google Cloud Storage")
    
    # Paths
    parser.add_argument("--raw-dir", type=str, default="matsynth_raw", help="Directory for raw export (or source for cleaning if --skip-export)")
    parser.add_argument("--clean-dir", type=str, default="matsynth_clean", help="Directory for cleaned data")
    
    # GCS options
    parser.add_argument("--bucket", type=str, default="main-testing", help="GCS bucket name")
    parser.add_argument("--prefix", type=str, default="raw", help="GCS prefix (default 'raw' for raw export, 'clean' if cleaning)")
    
    # Export options
    parser.add_argument("--limit", type=int, default=100, help="Max materials to export")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--skip-export", action="store_true", help="Skip export step (use existing raw-dir for cleaning)")
    
    # Clean options
    parser.add_argument("--resize", type=int, default=2048, help="Resize cleaned images")
    parser.add_argument("--flatten", action="store_true", help="Flatten cleaned output")
    parser.add_argument("--manifest", type=str, default="manifest.json", help="Manifest file name in clean dir")
    
    args = parser.parse_args()
    
    # Logic Flow
    
    # Case 1: Skip Export -> Clean existing raw data from disk
    if args.skip_export:
        if args.clean:
            print("Cleaning dataset from existing raw directory...")
            manifest_path = Path(args.clean_dir) / args.manifest
            clean_dataset_logic(
                src_dir=Path(args.raw_dir),
                dst_dir=Path(args.clean_dir),
                flatten=args.flatten,
                require_all=False,
                link=False,
                overwrite=True,
                dry_run=False,
                verbose=True,
                keep_ext=False,
                allowed_exts=ALLOWED_EXTENSIONS,
                manifest_path=manifest_path,
                resize=args.resize
            )
            if args.gcs:
                print("Uploading cleaned dataset to GCS...")
                upload_folder_to_gcs(args.clean_dir, args.bucket, args.prefix)
        else:
            print("Nothing to do. --skip-export used without --clean.")
        return

    # Case 2: Export + Clean -> Stream, clean in memory, save to dst (local or GCS)
    if args.clean:
        # Determine destination
        if args.gcs:
            # Stream -> Clean -> GCS
            # Default prefix for clean data if not specified? 
            # User might pass --prefix raw, but if cleaning, maybe we want 'clean'?
            # Let's respect the user's prefix, but maybe warn or default differently?
            # The prompt said "Have an option to export data (cleaned if using flag) to GCS"
            # So if --clean and --gcs, we write cleaned data to GCS.
            print("Streaming and cleaning dataset directly to GCS...")
            stream_and_clean(
                split=args.split,
                limit=args.limit,
                start=args.start,
                bucket_name=args.bucket,
                prefix=args.prefix,
                resize=args.resize,
                flatten=args.flatten
            )
        else:
            # Stream -> Clean -> Local
            print("Streaming and cleaning dataset to local directory...")
            manifest_path = Path(args.clean_dir) / args.manifest
            stream_and_clean(
                split=args.split,
                limit=args.limit,
                start=args.start,
                dst_dir=Path(args.clean_dir),
                resize=args.resize,
                flatten=args.flatten,
                manifest_path=manifest_path
            )
        return

    # Case 3: Export Only (Raw) -> Local or GCS
    if args.gcs:
        print("Exporting raw dataset directly to GCS...")
        export_gcs_direct(bucket_name=args.bucket, prefix=args.prefix, limit=args.limit, start=args.start)
    else:
        print("Exporting raw dataset to local directory...")
        export_local(dst=args.raw_dir, split=args.split, limit=args.limit, start=args.start)

if __name__ == "__main__":
    main()
