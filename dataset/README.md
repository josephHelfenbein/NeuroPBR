# Dataset

Contains scripts and resources for building and managing datasets.

- Includes raw PBR texture sets (from submodules like MatSynth).
- Stores generated multi-view IBL renders and ground-truth PBR maps.
- Used by the `renderer/` to generate synthetic training samples.

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
# Example: create MatSynth-like structure usable by the renderer
python dataset/clean_dataset.py \
  --src dataset/external/MatSynth \
  --dst dataset/testing \
  --require-all \
  --manifest dataset/testing/manifest.json

# If you want to keep original extensions instead of PNG:
python dataset/clean_dataset.py --src <src> --dst <dst> --keep-ext
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
