# Dataset

Contains scripts and resources for building and managing datasets.

- Pulls PBR texture sets on demand via the Hugging Face `datasets` library (e.g., MatSynth).
- Stores generated multi-view IBL renders and ground-truth PBR maps.
- Used by the `renderer/` to generate synthetic training samples.

## Exporting MatSynth via Hugging Face

The repository includes `dataset/export_matsynth.py`, which streams the public `gvecchio/MatSynth` dataset from Hugging Face and materializes it on disk—no external Git submodule required.

```bash
pip install datasets pillow
python dataset/export_matsynth.py \
  --dst dataset/matsynth_raw \
  --split train \
  --limit 500
```

Script highlights:

- Uses `datasets.load_dataset(..., streaming=True)` to avoid downloading the full archive.
- Exports the core channels (`basecolor`, `normal`, `roughness`, `metallic`) plus auxiliary maps when available.
- Writes per-material folders (`mat_00000`, `mat_00001`, …) so they can be passed directly into the cleaner or renderer pipelines.
- Adds the optional MatSynth metadata JSON per material when available.

## Exporting MatSynth to Google Cloud Storage

For cloud-based workflows, `dataset/export_matsynth_gcs.py` streams the MatSynth dataset and uploads it directly to a Google Cloud Storage bucket.

```bash
pip install google-cloud-storage datasets pillow
python dataset/export_matsynth_gcs.py
```

The script defaults to:
- Bucket: `main-testing`
- Prefix: `raw`
- Limit: 4000 materials

You can modify the `main()` function arguments in the script to change these defaults. It uploads the following maps as PNGs:
- `basecolor`, `normal`, `roughness`, `metallic`
- `diffuse`, `specular`, `displacement`, `opacity`, `blend_mask`

## Cleaning and Normalizing PBR Texture Sets

Use the Python script `dataset/clean_dataset.py` to scan a source directory recursively, detect common PBR map files, and create a cleaned dataset containing only the supported maps:

- albedo
- metallic
- roughness
- normal

The script normalizes names and optionally flattens output. By default it writes per-material folders with fixed PNG filenames:

- `albedo.png`
- `metallic.png`
- `roughness.png`
- `normal.png`

Use `--keep-ext` if you prefer to preserve original file extensions and skip PNG conversion. The cleaner supports copying or creating symlinks and can generate a manifest JSON.

### Quick start

```bash
cd dataset

# Create & activate a virtual environment (Linux/WSL2)
python3 -m venv .venv
source .venv/bin/activate
pip install datasets pillow

# 1) Export MatSynth locally (see section above)
python export_matsynth.py --dst matsynth_raw --limit 500

# 2) Clean and normalize the exported materials for the renderer
python clean_dataset.py \
  --src matsynth_raw \
  --dst matsynth_clean \
  --require-all \
  --manifest matsynth_clean/manifest.json

# Optional: keep original extensions instead of PNG
python clean_dataset.py --src <src> --dst <dst> --keep-ext
```

### CLI options

- `--src`: Source directory to scan recursively
- `--dst`: Destination directory for the cleaned dataset
- `--flatten`: Flatten output into `dst` as `<material>_<map>.<ext>` instead of per-material folders
- `--require-all`: Only include materials that have all four maps
- `--link`: Create symlinks instead of copying files
- `--overwrite`: Overwrite destination files if they exist
- `--dry-run`: Print planned operations without writing files
- `--verbose`: Extra logging
- `--ext`: Whitelist of file extensions to consider (default supports common image formats)
- `--keep-ext`: Keep original file extensions; otherwise the cleaner converts outputs to PNG and fixes names
- `--manifest`: Optional path to write a JSON manifest of included materials and map paths

### Detection heuristics

The script matches filenames using common synonyms:

- albedo: `albedo`, `basecolor`, `base colour`, `diffuse`
- metallic: `metallic`, `metalness`, `metal`, `mtl`, `met`
- roughness: `roughness`, `rough`, `rgh`
- normal: `normal`, `normals`, `nrm`, `nor`, `n`, `normalgl`

Notes:

- The cleaner does not change PBR semantics (e.g., glossiness↔roughness, specular workflows). It only renames and optionally converts image containers to PNG.
