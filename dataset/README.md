# Dataset

Contains scripts and resources for building and managing datasets.

- Pulls PBR texture sets on demand via the Hugging Face `datasets` library (e.g., MatSynth).
- Stores generated multi-view IBL renders and ground-truth PBR maps.
- Used by the `renderer/` to generate synthetic training samples.

## Processing Dataset (Export, Clean, Upload)

The `dataset/process_dataset.py` script consolidates exporting, cleaning, and uploading functionality into a single tool.

It can:
1.  **Export** the MatSynth dataset from Hugging Face to a local directory or directly to Google Cloud Storage (GCS).
2.  **Clean** and normalize the dataset (standardizing names to albedo, metallic, roughness, normal).
3.  **Upload** the cleaned dataset to GCS.

**Note:** When using the `--clean` flag, the script processes images **in-memory**. It streams data from Hugging Face, cleans/resizes it in RAM, and writes *only* the final cleaned files to disk or GCS. Raw files are not saved to disk unless you explicitly export them without cleaning.

### Usage Examples

**Export raw data locally:**
```bash
python dataset/process_dataset.py \
  --raw-dir dataset/matsynth_raw \
  --limit 500 \
  --start 0
```

**Export raw data directly to GCS:**
```bash
python dataset/process_dataset.py \
  --gcs \
  --bucket main-testing \
  --prefix raw \
  --limit 4000
```

**Stream, clean in-memory, and save locally:**
```bash
python dataset/process_dataset.py \
  --clean \
  --clean-dir dataset/matsynth_clean \
  --limit 100
```

**Stream, clean in-memory, and upload directly to GCS:**
```bash
python dataset/process_dataset.py \
  --clean \
  --gcs \
  --bucket main-testing \
  --prefix clean_data
```

**Clean existing raw data on disk:**
```bash
python dataset/process_dataset.py \
  --clean \
  --skip-export \
  --raw-dir dataset/matsynth_raw \
  --clean-dir dataset/matsynth_clean
```

### CLI Options

- `--clean`: Enable cleaning step.
- `--gcs`: Enable GCS upload (direct export if not cleaning, or upload after cleaning).
- `--raw-dir`: Directory for raw export (default: `matsynth_raw`).
- `--clean-dir`: Directory for cleaned data (default: `matsynth_clean`).
- `--bucket`: GCS bucket name.
- `--prefix`: GCS prefix.
- `--limit`: Max materials to export.
- `--start`: Start index.
- `--split`: Dataset split (default: `train`).
- `--skip-export`: Skip the export step (use existing data in `--raw-dir`).
- `--resize`: Resize cleaned images (default: 2048).
- `--flatten`: Flatten cleaned output.
- `--manifest`: Manifest file name.

## Cleaning and Normalizing PBR Texture Sets

The `dataset/clean_dataset.py` script is still available for standalone cleaning tasks, though `process_dataset.py` wraps its functionality. It scans a source directory recursively, detects common PBR map files, and creates a cleaned dataset containing only the supported maps:

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

### CLI options (clean_dataset.py)

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

### Detection heuristics (clean_dataset.py)

The script matches filenames using common synonyms:

- albedo: `albedo`, `basecolor`, `base colour`, `diffuse`
- metallic: `metallic`, `metalness`, `metal`, `mtl`, `met`
- roughness: `roughness`, `rough`, `rgh`
- normal: `normal`, `normals`, `nrm`, `nor`, `n`, `normalgl`

Notes:

- The cleaner does not change PBR semantics (e.g., glossiness↔roughness, specular workflows). It only renames and optionally converts image containers to PNG.
