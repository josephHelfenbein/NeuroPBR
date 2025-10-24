# NeuroPBR



## Cloning the NeuroPBR Repository

NeuroPBR uses Git submodules and Git LFS for large PBR datasets. To avoid stalling on LFS objects, follow these steps:

### 1. Clone the repository without downloading large LFS files immediately

**Linux / macOS:**
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/YourUser/NeuroPBR.git
```
**Windows (PowerShell):**
```bash
$env:GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/YourUser/NeuroPBR.git
Remove-Item Env:\GIT_LFS_SKIP_SMUDGE
```
This clones the repository quickly by skipping automatic download of LFS files.

### 2. Initialize and update submodules
```bash
cd NeuroPBR
git submodule update --init --recursive
```
This fetches all submodules (dataset/external/MatSynth, etc.) with pointer files only.

### 3. Download full LFS content (optional)
If you need the full dataset for training or rendering:
```bash
git lfs pull --recursive
```
- Can be run from the repo root to fetch LFS objects for all submodules.
- This will download the actual .parquet PBR dataset files tracked by LFS.