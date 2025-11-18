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
This fetches all code submodules (shaders, libs, etc.) with pointer files only.

### 3. Download full LFS content (optional)
If you need the full set of LFS-tracked binaries (HDRIs, cached builds, etc.):
```bash
git lfs pull --recursive
```
- Can be run from the repo root to fetch LFS objects for all submodules.
- Skip this step if you only need source + scripts.

## Fetching MatSynth Textures via Hugging Face

The repository no longer relies on an external MatSynth submodule. Instead, use the included exporter powered by the Hugging Face `datasets` library:

```bash
pip install datasets pillow
python dataset/export_matsynth.py \
	--dst dataset/matsynth_raw \
	--split train \
	--limit 500
	End-to-end pipeline for generating synthetic physically based rendering (PBR) datasets and training a multi-view GAN to reconstruct material properties from rendered images.

	## Repository Layout

	- `dataset/` – Hugging Face powered exporters, cleaners, and docs for preparing PBR materials.
	- `renderer/` – CUDA/C++ renderer that produces paired dirty/clean views + metadata for training.
	- `training/` – PyTorch training stack (multi-view encoder, ViT fusion, UNet decoder, GAN losses).
	- `mobile_app/` – Experimental mobile front-ends (not covered here).

	## Prerequisites

	- Windows 10/11 or Linux with an NVIDIA GPU (CUDA-capable, 8 GB VRAM or more recommended).
	- CUDA Toolkit + CMake 3.18+ + Visual Studio 2022 (or clang/gcc + Ninja) for the renderer.
	- Python 3.8+ for dataset scripts and the training pipeline.

	---

	## 1. Prepare PBR Materials (`dataset/`)

	1. **Create an isolated Python environment and install dependencies.**

	```powershell
	cd dataset
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install datasets pillow
	```

	2. **Stream MatSynth via Hugging Face and cache it locally.**

	```powershell
	python export_matsynth.py --dst matsynth_raw --split train --limit 500
	```

	Adjust `--split`, `--limit`, or `--save-metadata False` to control how much you pull.

	3. **Normalize map names and ensure every material has the required channels.**

	```powershell
	python clean_dataset.py `
		--src matsynth_raw `
		--dst matsynth_clean `
		--require-all `
		--manifest matsynth_clean/manifest.json
	```

	Useful flags:

	- `--keep-ext` preserves original formats instead of converting to PNG.
	- `--flatten` writes `<material>_<map>.png` into a single directory if you prefer a non-nested layout.

	See `dataset/README.md` for detailed CLI descriptions and troubleshooting tips.

	---

	## 2. Build the Renderer (`renderer/`)

	1. **Configure + build (Visual Studio 2022 example).**

	```powershell
	cd renderer
	cmake -G "Visual Studio 17 2022" -A x64 -T host=x64 -S . -B build
	cmake --build build --config Release --parallel
	```

	Ninja/Make-based workflows are also supported (see `renderer/README.md`).

	2. **Generate synthetic renders.**

	```powershell
	cd renderer
	.\bin\Release\neuropbr_renderer.exe ..\dataset\matsynth_clean 2000
	```

	Arguments: `<textures_dir> <num_samples>`. The renderer automatically creates `output/clean`, `output/dirty`, and `output/render_metadata.json`, writing three views per sample with randomized lighting and artifacts.

	Ensure the HDRI, smudge, and lens flare assets exist under `renderer/assets/` before rendering.

	---

	## 3. Train the Model (`training/`)

	1. **Install training dependencies.**

	```powershell
	cd training
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	```

	2. **Launch training using the renderer outputs.**

	```powershell
	python train.py `
		--input-dir ..\renderer\output `
		--output-dir ..\dataset\matsynth_clean `
		--batch-size 2
	```

	Key options:

	- `--input-dir / --output-dir / --metadata-path` let you point to any folder layout.
	- `--use-dirty` trains with dirty renders instead of clean ones (clean is the default input).
	- The dataloader downsamples both renders and targets to 1024×1024 by default, so you can keep full-res assets on disk without preprocessing.
	- `--device {auto|cuda|cuda:0|cpu}` forces the accelerator if auto-detection doesn't pick the GPU you expect.
	- Preset configs like `--config quick_test` or `--config lightweight` adjust model/compute tradeoffs.

	Refer to `training/README.md` and `TRAINING_GUIDE.md` for the loss breakdown, advanced configs, and troubleshooting steps.

	---

	## Additional Resources

	- `dataset/README.md` – Deep dive on exporters, cleaning heuristics, and CLI options.
	- `renderer/README.md` – Detailed build instructions and asset requirements.
	- `training/README.md` & `TRAINING_GUIDE.md` – Model architecture, configs, and evaluation metrics.
