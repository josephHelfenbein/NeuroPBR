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

The script normalizes names and optionally flattens output. It supports copying or creating symlinks and can generate a manifest JSON.

### Quick start

```bash
python dataset/clean_dataset.py --src dataset/external/MatSynth --dst dataset/cleaned --require-all --manifest dataset/cleaned/manifest.json
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
- `--manifest`: Optional path to write a JSON manifest of included materials and map paths

### Detection heuristics

The script matches filenames using common synonyms:

- albedo: `albedo`, `basecolor`, `base colour`, `diffuse`
- metallic: `metallic`, `metalness`, `metal`, `mtl`, `met`
- roughness: `roughness`, `rough`, `rgh`
- normal: `normal`, `normals`, `nrm`, `nor`, `n`, `normalgl`

Note: No conversions are performed (e.g., glossiness to roughness, or specular workflows). If you need those, convert upstream before cleaning.
