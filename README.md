# NeuroPBR

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

## Cloning the NeuroPBR Repository

**Linux / macOS:**
```bash
git clone https://github.com/YourUser/NeuroPBR.git
cd NeuroPBR
git submodule update --init --recursive
```
**Windows (PowerShell):**
```bash
git clone https://github.com/YourUser/NeuroPBR.git
cd NeuroPBR
git submodule update --init --recursive
```

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
- `--render-curriculum {0|1|2}` picks clean-only, dataset-balanced clean+dirty, or dirty-only inputs (`--use-dirty` remains a shortcut for `2`).
- The dataloader downsamples both renders and targets to 1024×1024 by default, so you can keep full-res assets on disk without preprocessing.
- `--device {auto|cuda|cuda:0|cpu}` forces the accelerator if auto-detection doesn't pick the GPU you expect.
- Preset configs like `--config quick_test` or `--config lightweight` adjust model/compute tradeoffs.

Refer to `training/README.md` and `TRAINING_GUIDE.md` for the loss breakdown, advanced configs, and troubleshooting steps.

---

## Additional Resources

- `dataset/README.md` – Deep dive on exporters, cleaning heuristics, and CLI options.
- `renderer/README.md` – Detailed build instructions and asset requirements.
- `training/README.md` & `TRAINING_GUIDE.md` – Model architecture, configs, and evaluation metrics.
