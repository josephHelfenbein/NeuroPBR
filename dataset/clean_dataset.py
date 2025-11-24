"""
Clean and normalize PBR texture sets.

Recursively finds albedo, metallic, roughness, and normal maps, then writes a
consistently named dataset (copy or symlink). Supports per-material folders or
flattened outputs.

Update: by default, outputs fixed filenames per material folder using PNG
extension — "albedo.png", "metallic.png", "roughness.png", "normal.png" — to
match common renderer expectations. You can keep original extensions with
"--keep-ext".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from PIL import Image
except Exception:
    Image = None


TARGET_MAPS: Tuple[str, ...] = ("albedo", "metallic", "roughness", "normal")
ALLOWED_EXTENSIONS: Set[str] = {".png", ".jpg",
                                ".jpeg", ".exr", ".tif", ".tiff", ".webp"}


def debug(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def slugify(text: str) -> str:
    """Create a filesystem-friendly slug for a material name."""
    text = text.strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^a-z0-9\-]+", "", text)
    text = re.sub(r"\-+", "-", text)
    return text or "material"


_MAP_PATTERNS: Dict[str, List[re.Pattern]] = {
    # Albedo / BaseColor
    "albedo": [
        re.compile(r"\balbedo\b", re.IGNORECASE),
        re.compile(r"\bbase\s*color\b|\bbasecolor\b|\bbc\b", re.IGNORECASE),
        re.compile(r"\bbase\s*colour\b|\bbasecolour\b", re.IGNORECASE),
        re.compile(r"\bdiffuse\b|\bdiff\b", re.IGNORECASE),
    ],
    # Metallic / Metalness
    "metallic": [
        re.compile(r"\bmetallic\b|\bmetalness\b|\bmetalic\b", re.IGNORECASE),
        re.compile(r"\bmetal\b|\bmet\b|\bmtl\b", re.IGNORECASE),
    ],
    # Roughness
    "roughness": [
        re.compile(r"\broughness\b|\brough\b|\brgh\b", re.IGNORECASE),
    ],
    # Normal
    "normal": [
        re.compile(r"\bnormal\b|\bnormals\b|\bnrm\b|\bnor\b", re.IGNORECASE),
        re.compile(r"(?:^|[_\-\.])n(?:[_\-\.]|$)", re.IGNORECASE),
        re.compile(r"\bnormalgl\b|\bnorm\s*gl\b", re.IGNORECASE),
    ],
}


def strip_udim_token(name_no_ext: str) -> str:
    """Remove UDIM-like tokens (e.g., _1001, .1001) to avoid misclassification."""
    return re.sub(r"([_\.])1\d{3}$", "", name_no_ext)


def guess_map_type(filename: str) -> Optional[str]:
    """Guess the PBR map type from a filename using heuristics.

    Returns one of TARGET_MAPS or None if no match.
    """
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


def find_materials(
    src_dir: Path,
    allowed_exts: Set[str],
    verbose: bool,
) -> List[MaterialEntry]:
    materials: Dict[Path, MaterialEntry] = {}

    for root, _dirs, files in os.walk(src_dir):
        root_path = Path(root)
        parts = root_path.parts
        if any(p.startswith(".") for p in parts):
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
                entry = MaterialEntry(
                    material_root=root_path, material_slug=material_slug)
                materials[root_path] = entry

            # Prefer first match; if duplicate map in same folder, keep the shorter filename as heuristic
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


def convert_to_png(src: Path, dst: Path, map_type: str, verbose: bool) -> bool:
    """Convert an image file to PNG at the destination path.

    Returns True on success, False if conversion could not be performed.
    """
    if Image is None:
        debug("Pillow not available; cannot convert to PNG.", verbose)
        return False
    try:
        with Image.open(src) as im:
            # Normalize mode by map type: albedo/normal as RGB, roughness/metallic as L
            if map_type in ("albedo", "normal"):
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                else:
                    # If RGBA, drop alpha to keep a stable 3-channel layout
                    if im.mode == "RGBA":
                        im = im.convert("RGB")
            else:  # roughness / metallic
                # Preserve single-channel if possible; otherwise convert to L
                if im.mode not in ("L", "I;16"):
                    im = im.convert("L")
            # Ensure parent exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Resize to 1024x1024
            im = im.resize((1024, 1024), Image.LANCZOS)
            
            im.save(dst, format="PNG", optimize=True)
        return True
    except Exception as e:  # pragma: no cover - best-effort conversion
        debug(f"PNG conversion failed for {src}: {e}", verbose)
        return False


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean and normalize a PBR texture dataset.")
    p.add_argument("--src", required=True, type=Path,
                   help="Source directory to scan recursively")
    p.add_argument("--dst", required=True, type=Path,
                   help="Destination directory for cleaned dataset")
    p.add_argument("--flatten", action="store_true",
                   help="Do not create per-material subfolders; flatten into <slug>_<map><ext>")
    p.add_argument("--require-all", action="store_true",
                   help="Require all four maps for a material to be included")
    p.add_argument("--link", action="store_true",
                   help="Create symlinks instead of copying files")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing destination files if present")
    p.add_argument("--dry-run", action="store_true",
                   help="Show planned operations without writing files")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument(
        "--keep-ext",
        action="store_true",
        help="Keep original file extensions instead of converting outputs to .png",
    )
    p.add_argument(
        "--ext",
        nargs="*",
        default=sorted(ALLOWED_EXTENSIONS),
        help="Allowed file extensions to consider (default: common image formats)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write a manifest JSON of included materials",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)

    src_dir: Path = args.src
    dst_dir: Path = args.dst
    flatten: bool = bool(args.flatten)
    require_all: bool = bool(args.require_all)
    link: bool = bool(args.link)
    overwrite: bool = bool(args.overwrite)
    dry_run: bool = bool(args.dry_run)
    verbose: bool = bool(args.verbose)
    keep_ext: bool = bool(args.keep_ext)
    allowed_exts: Set[str] = {e.lower() if e.startswith(
        ".") else f".{e.lower()}" for e in args.ext}
    manifest_path: Optional[Path] = args.manifest

    if not src_dir.exists() or not src_dir.is_dir():
        print(f"ERROR: --src does not exist or is not a directory: {src_dir}")
        return 2

    ensure_dir(dst_dir)

    debug(f"Scanning source: {src_dir}", verbose)
    materials = find_materials(src_dir, allowed_exts, verbose)

    included: List[Dict[str, str]] = []
    total_candidates = 0
    total_included = 0

    for entry in sorted(materials, key=lambda m: m.material_slug):
        total_candidates += 1
        present_maps = set(entry.files_by_map.keys())
        if require_all and not all(m in present_maps for m in TARGET_MAPS):
            debug(
                f"Skip '{entry.material_slug}': missing maps {set(TARGET_MAPS) - present_maps}",
                verbose,
            )
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

        record: Dict[str, str] = {"material": entry.material_slug}

        for map_type in maps_to_use:
            src_path = entry.files_by_map[map_type]
            ext = src_path.suffix
            # Destination naming: default to fixed .png filenames for consistency
            if flatten:
                dst_path = dest_dir / (
                    f"{entry.material_slug}_{map_type}.png" if not keep_ext else
                    f"{entry.material_slug}_{map_type}{ext}"
                )
            else:
                dst_path = dest_dir / (
                    f"{map_type}.png" if not keep_ext else f"{map_type}{ext}"
                )

            if dry_run:
                action = "convert" if not keep_ext else (
                    "link" if link else "copy")
                print(f"Would {action} {src_path} -> {dst_path}")
            else:
                if keep_ext:
                    copy_or_link(src_path, dst_path, link=link,
                                 overwrite=overwrite)
                else:
                    # Try to convert; if conversion fails, fall back to copy with original ext
                    success = convert_to_png(
                        src_path, dst_path, map_type, verbose)
                    if not success:
                        # Fallback path retains original extension to avoid mismatch
                        fallback_dst = dst_path.with_suffix(ext)
                        copy_or_link(src_path, fallback_dst, link=link,
                                     overwrite=overwrite)
                        # Update record to actual file written
                        dst_path = fallback_dst

            record[map_type] = str(dst_path)

        included.append(record)
        total_included += 1

    if manifest_path is not None:
        if dry_run:
            print(f"Would write manifest: {manifest_path}")
        else:
            ensure_dir(manifest_path.parent)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"materials": included}, f, indent=2)
        debug(f"Manifest entries: {len(included)}", verbose)

    print(
        f"Materials scanned: {total_candidates}, included: {total_included} (dst: {dst_dir})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
