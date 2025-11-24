# NeuroPBR

End-to-end pipeline for generating synthetic physically based rendering (PBR) datasets and training a multi-view GAN to reconstruct material properties from rendered images.

## Repository Layout

- `dataset/` – Hugging Face powered exporters, cleaners, and docs for preparing PBR materials.
- `renderer/` – CUDA/C++ renderer that produces paired dirty/clean views + metadata for training.
- `training/` – PyTorch training stack (multi-view encoder, ViT fusion, UNet decoder, GAN losses).
- `mobile_app/` – Experimental mobile front-ends (not covered here).

## Prerequisites

- **Linux or WSL2 (Windows Subsystem for Linux)** is required for the training pipeline (due to `torch.compile` and `triton` dependencies).
- NVIDIA GPU (CUDA-capable, 8 GB VRAM or more recommended).
- CUDA Toolkit + CMake 3.18+ + GCC/Clang (for renderer).
- Python 3.10+ for dataset scripts and the training pipeline.

---

## Cloning the NeuroPBR Repository

**Linux / WSL2:**
```bash
git clone https://github.com/YourUser/NeuroPBR.git
cd NeuroPBR
git submodule update --init --recursive
```

## 1. Prepare PBR Materials (`dataset/`)

1. **Create an isolated Python environment and install dependencies.**

```bash
cd dataset
python3 -m venv .venv
source .venv/bin/activate
pip install datasets pillow
```

2. **Stream MatSynth via Hugging Face and cache it locally.**

```bash
python export_matsynth.py --dst matsynth_raw --split train --limit 500
```

Adjust `--split`, `--limit`, or `--save-metadata False` to control how much you pull.

3. **Normalize map names and ensure every material has the required channels.**

```bash
python clean_dataset.py \
	--src matsynth_raw \
	--dst matsynth_clean \
	--require-all \
	--manifest matsynth_clean/manifest.json
```

Useful flags:
- `--keep-ext` preserves original formats instead of converting to PNG.

See `dataset/README.md` for detailed CLI descriptions and troubleshooting tips.

---

## 2. Build the Renderer (`renderer/`)

1. **Configure + build (Linux/WSL2).**

```bash
cd renderer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

2. **Generate synthetic renders.**

```bash
cd renderer
./bin/neuropbr_renderer ../dataset/matsynth_clean 2000
```

Arguments: `<textures_dir> <num_samples>`. The renderer automatically creates `output/clean`, `output/dirty`, and `output/render_metadata.json`, writing three views per sample with randomized lighting and artifacts.

---

## 3. Train the Model (`training/`)

1. **Install training dependencies.**

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Launch training using the renderer outputs.**

```bash
python train.py \
	--input-dir ../renderer/output \
	--output-dir ../dataset/matsynth_clean \
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

## 4. GPU Optimization & Performance

The training pipeline automatically detects your hardware and applies the best optimizations:

- **Automatic `torch.compile`**: On PyTorch 2.0+ and modern GPUs (Ampere/Hopper), models are compiled for up to 30% faster training.
- **Mixed Precision (AMP)**: Automatically selects BFloat16 (Ampere+) or Float16 (Volta/Turing).
- **TensorFloat-32 (TF32)**: Enabled by default on RTX 30/40 series and A100/H100.
- **Memory Layout**: Models are converted to `channels_last` format for better tensor core utilization.

**Manual Controls (Environment Variables):**

- `USE_TORCH_COMPILE=false`: Disable model compilation if you encounter bugs.
- `USE_TORCH_COMPILE=true`: Force compilation on unsupported hardware.
- `IS_SPOT_INSTANCE=true`: Use faster compilation mode (`reduce-overhead`) to save time on short-lived instances.

---

## Additional Resources

- `dataset/README.md` – Deep dive on exporters, cleaning heuristics, and CLI options.
- `renderer/README.md` – Detailed build instructions and asset requirements.
- `training/README.md` & `TRAINING_GUIDE.md` – Model architecture, configs, and evaluation metrics.
