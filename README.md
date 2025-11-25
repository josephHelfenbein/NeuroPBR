[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


[contributors-shield]: https://img.shields.io/github/contributors/josephHelfenbein/NeuroPBR.svg?style=for-the-badge
[contributors-url]: https://github.com/josephHelfenbein/NeuroPBR/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/josephHelfenbein/NeuroPBR.svg?style=for-the-badge
[forks-url]: https://github.com/josephHelfenbein/NeuroPBR/network/members
[stars-shield]: https://img.shields.io/github/stars/josephHelfenbein/NeuroPBR.svg?style=for-the-badge
[stars-url]: https://github.com/josephHelfenbein/NeuroPBR/stargazers
[issues-shield]: https://img.shields.io/github/issues/josephHelfenbein/NeuroPBR.svg?style=for-the-badge
[issues-url]: https://github.com/josephHelfenbein/NeuroPBR/issues
[license-shield]: https://img.shields.io/github/license/josephHelfenbein/NeuroPBR.svg?style=for-the-badge
[license-url]: https://github.com/josephHelfenbein/NeuroPBR/blob/master/LICENSE.txt

<!-- PROJECT LOGO -->
<br />
<div align="center">

<a href="https://github.com/josephHelfenbein/NeuroPBR">
    <img src="mobile_app/assets/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">NeuroPBR</h3>

<p align="center">
    An end-to-end system for reconstructing PBR materials from handheld photos. Features a custom synthetic data renderer, a multi-view deep learning model, and a mobile app for on-device inference.
	<br />
    <br />
    <a href="https://github.com/josephHelfenbein/NeuroPBR/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/josephHelfenbein/NeuroPBR/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## About the Project

NeuroPBR is an end-to-end system for digitizing real-world materials into high-quality PBR (Physically Based Rendering) textures using an iPhone. It enables developers and artists to create professional-quality 3D materials using just an iPhone by combining:

1.  **Synthetic Data Generation**: A custom C++/CUDA renderer that produces photorealistic training pairs (clean PBR maps vs. artifact-heavy renders) from the MatSynth dataset.
2.  **Deep Learning Pipeline**: A multi-view fusion network (ResNet/UNet + Vision Transformer) trained to reconstruct albedo, normal, roughness, and metallic maps from just three imperfect photos.
3.  **Mobile Deployment**: An iOS app that runs a distilled "Student" model on-device via Core ML, featuring a real-time Metal-based PBR previewer for instant feedback.

This repository contains the complete stack: from dataset preparation and rendering to model training and mobile deployment.

## Repository Layout

- `dataset/` – Hugging Face powered exporters, cleaners, and docs for preparing PBR materials.
- `renderer/` – CUDA/C++ renderer that produces paired dirty/clean views + metadata for training.
- `training/` – PyTorch training stack (multi-view encoder, ViT fusion, UNet decoder, GAN losses).
- `mobile_app/` – Flutter iOS app for capture, on-device inference (Core ML), and Metal-based PBR preview.

## Prerequisites

- **Linux or WSL2 (Windows Subsystem for Linux)** is required for the training pipeline (due to `torch.compile` and `triton` dependencies).
- NVIDIA GPU (CUDA-capable, 16 GB VRAM or more recommended).
- CUDA Toolkit + CMake 3.18+ + GCC/Clang (for renderer).
- Python 3.10+ for dataset scripts and the training pipeline.

---

## Cloning the NeuroPBR Repository

**Linux / WSL2:**
```bash
git clone https://github.com/josephHelfenbein/NeuroPBR.git
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
- The dataloader loads images at native 2048×2048 resolution.
- `--device {auto|cuda|cuda:0|cpu}` forces the accelerator if auto-detection doesn't pick the GPU you expect.
- Preset configs like `--config quick_test` or `--config lightweight` adjust model/compute tradeoffs.

Refer to `training/README.md` for the loss breakdown, advanced configs, and troubleshooting steps.

---

## 4. Mobile App (`mobile_app/`)

The mobile application brings the reconstruction pipeline to the edge:

1.  **Capture**: Guides users to take 3 specific photos of a surface.
2.  **Inference**: Runs a distilled "Student" model via Core ML directly on the device.
3.  **Preview**: Visualizes the material using a custom C++/Metal renderer (ported from the main CUDA renderer).

See `mobile_app/README.md` for setup and build instructions.

---

## 5. GPU Optimization & Performance

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
- `training/README.md` – Model architecture, configs, and evaluation metrics.
- `mobile_app/README.md` – iOS app setup, architecture, and usage.
