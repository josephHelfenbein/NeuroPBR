# NeuroPBR Training

Complete guide for training the multi-view fusion GAN that reconstructs PBR textures from three rendered views.

---

## Overview

NeuroPBR trains a deep model that:

1. Consumes **three rendered views** (clean or dirty) per material
2. Encodes each with a shared ResNet/UNet backbone
3. Fuses view features via a **Vision Transformer cross-view block**
4. Decodes to **four 2048×2048 PBR maps** (albedo, roughness, metallic, normal)
5. Improves realism with **optional GAN losses** plus reconstruction terms


### Key Features

- Multi-view fusion with ViT attention  
- Flexible loss system (L1 / SSIM / Normal / GAN)  
- Mixed precision + gradient clipping  
- Automatic train/val split with reproducible seeding  
- Checkpointing (latest, best, periodic) and resume  
- TensorBoard + optional WandB logging  
- Scriptable inference utility for trained checkpoints

---

## Quick Start

### 0. System Requirements
**Linux or WSL2 is required.**
The training pipeline uses `torch.compile` and `triton`, which are not fully supported on Windows. Please run this in a WSL2 environment (Ubuntu recommended) with the NVIDIA Container Toolkit or proper CUDA driver forwarding.

### 1. Install Dependencies
```bash
cd training
python3 -m venv .venv
source .venv/bin/activate

# For macOS (Apple Silicon/Intel) - includes coremltools
pip install -r requirements_macos.txt

# For Linux - includes triton
pip install -r requirements_linux.txt
```
Optional extras (uncomment in the respective requirements file): `wandb`, `opencv-python`, `matplotlib`.

### 2. Verify Dataset Layout
```

input/
├── clean/sample_0000/{0,1,2}.png
├── dirty/sample_0000/{0,1,2}.png
└── render_metadata.json

output/
└── material_0/{albedo,roughness,metallic,normal}.png
```
`render_metadata.json` must map `sample_XXXX` folders to `material_name` directories. Both directories are specified in the arguments when running training.

### 3. Basic Training
```bash
python train.py --input-dir ./data/input --output-dir ./data/output
# Explicit CUDA selection
python train.py --input-dir ./data/input --output-dir ./data/output --device cuda

# Mixed clean/dirty renders (curriculum 1)
python train.py --input-dir ./data/input --output-dir ./data/output --render-curriculum 1

# Save checkpoints to a custom directory
python train.py --input-dir ./data/input --output-dir ./data/output --checkpoint-dir ./my_checkpoints
```

### 4. Alternate Presets
```bash
# Quick smoke test
python train.py --config quick_test --input-dir ./data/input --output-dir ./data/output --batch-size 2

# Lightweight / no GAN
python train.py --config lightweight --input-dir ./data/input --output-dir ./data/output

# Custom config file
python train.py --config configs/my_config.py --input-dir ./data/input --output-dir ./data/output
```

### 5. Resume Training
```bash
python train.py --resume checkpoints/checkpoint_epoch_0050.pth
```

---

## Dataset Setup

### Structure
```
input/
├── clean/sample_0000/{0,1,2}.png
├── dirty/sample_0000/{0,1,2}.png // optional
└── render_metadata.json

output/
└── material_0/{albedo,roughness,metallic,normal}.png
```
- **Input renders**: three views per sample (dirty = artifact-heavy, clean = pristine)
- **Output maps**: RGB/gray PBR textures per material
- **Metadata**: ensures renders map to the correct material folder

### Metadata Example
```json
{
  "sample_0000": "wood_oak",
  "sample_0001": "metal_rusty"
}
```
Location: `{input_dir}/render_metadata.json`.

### Render Curriculum (Clean ↔ Dirty)
```python
config.data.render_curriculum = 0  # default clean-only curriculum
config.data.render_curriculum = 1  # Match on-disk clean/dirty mix each epoch
config.data.render_curriculum = 2  # dirty-only curriculum
```
Use `--render-curriculum {0|1|2}` on the CLI to override per run. 
The legacy `--use-dirty` flag still works and simply maps to curriculum `2` for backwards compatibility.

| Curriculum | Clean Renders | Dirty Renders | Notes |
| --- | --- | --- | --- |
| `0` | 100% | 0% | Default clean-only training |
| `1` | Mirrors dataset | Mirrors dataset | Mix follows whatever ratio exists on disk (clean entry included once, dirty once) |

When a sample exists in both folders it is queued twice (once per source); otherwise it appears only where present,
so the loader respects the natural clean/dirty balance without duplicating the smaller pool.
| `2` | 0% | 100% | Use for dirty-only runs (e.g., student training) |

### Automatic Train/Val Split
```python
config.data.val_ratio = 0.1  # default 90/10 split
config.training.seed = 42    # reproducible shuffling
```
Samples are shuffled once per run and split deterministically with the seed.

### On-the-Fly Resizing
Both inputs and targets are loaded at native 2048×2048 resolution (default `config.data.image_size`). No downscaling is applied by default.

### Dataset Verification Snippet
```bash
python - <<'PY'
from utils.dataset import PBRDataset
from train_config import get_default_config
cfg = get_default_config()
ds = PBRDataset(
    input_dir='./data/input',
    output_dir='./data/output',
    metadata_path='./data/input/render_metadata.json',
    transform_mean=cfg.transform.mean,
    transform_std=cfg.transform.std,
    image_size=cfg.data.image_size,
  use_dirty=cfg.data.use_dirty_renders,
  curriculum_mode=cfg.data.render_curriculum
)
print('Total samples:', len(ds))
inputs, targets = ds[0]
print('Input shape:', inputs.shape)
print('Target shape:', targets.shape)
PY
```

### Common Issues
| Symptom | Fix |
| --- | --- |
| `FileNotFoundError: render_metadata.json` | Create `{input_dir}/render_metadata.json` |
| `KeyError: sample_XXXX` | Ensure metadata lists every `sample_*` folder |
| "No samples found" | Verify `input/dirty` or `clean` folders and `output/material` folders contain all required PNGs |

---

## Configuration System

### Preset Configurations
| Name | Highlights | Use Case |
| --- | --- | --- |
| `default` | ResNet50 encoder, ViT depth 4, GAN starts epoch 5 | Production training |
| `quick_test` | ResNet18, ViT depth 2, 20 epochs | Debug / smoke tests |
| `lightweight` | ResNet50, no GAN | Fast baseline |
| `configs/high_quality.py` | ResNet101, deeper ViT | Highest fidelity |
| `configs/fast_iteration.py` | ResNet18, fewer epochs | Rapid experimentation |
| `configs/normal_focused.py` | Normal-heavy loss weights | Emphasize normals |

### Custom Config Template
```python
from train_config import TrainConfig

def get_config():
    cfg = TrainConfig()
    cfg.model.encoder_backbone = 'resnet101'
    cfg.model.transformer_depth = 6
    cfg.model.use_gan = True

    cfg.training.epochs = 200
    cfg.training.batch_size = 8

    cfg.loss.w_l1 = 1.0
    cfg.loss.w_ssim = 0.5
    cfg.loss.w_normal = 0.7

    cfg.optimizer.g_lr = 1e-4
    cfg.optimizer.d_lr = 4e-4
    cfg.optimizer.scheduler = 'cosine'
    return cfg
```
Run via `python train.py --config configs/my_config.py ...`.

---

## Loss Configuration

### Available Components
| Loss | Purpose | Key | Default |
| --- | --- | --- | --- |
| L1 | Per-pixel reconstruction | `w_l1` | 1.0 |
| SSIM | Structural similarity (albedo) | `w_ssim` | 0.3 |
| Normal | Angular error for normals | `w_normal` | 0.5 |
| GAN | Adversarial realism | `w_gan` | 0.05 |

### Per-Map Weights
```python
cfg.loss.w_albedo = 1.0
cfg.loss.w_roughness = 1.0
cfg.loss.w_metallic = 1.0
cfg.loss.w_normal_map = 1.0
```

### Recipes
- Emphasize normals → raise `w_normal`, `w_normal_map`
- Photographic albedo → bump `w_albedo`, `w_ssim`
- Disable GAN → set `cfg.model.use_gan = False`, `cfg.loss.w_gan = 0.0`

### GAN Modes
```python
cfg.loss.gan_loss_type = 'hinge'  # default
cfg.loss.gan_loss_type = 'bce'
```

---

## Model Architecture

### Encoder Options
```python
cfg.model.encoder_type = 'resnet'
cfg.model.encoder_backbone = 'resnet50'  # 18 / 34 / 50 / 101 / 152
cfg.model.encoder_stride = 1             # keep 2048 resolution

cfg.model.encoder_type = 'unet'
cfg.model.encoder_channels = [64,128,256,512,1024,2048]
```

### Transformer (Cross-View Fusion)
```python
cfg.model.transformer_dim = 2048
cfg.model.transformer_num_heads = 32
cfg.model.transformer_depth = 4
cfg.model.transformer_mlp_ratio = 4
```
Adjust heads/depth for lighter or heavier configs.

### Discriminator
```python
cfg.model.discriminator_type = 'configurable'
cfg.model.discriminator_n_layers = 6
cfg.model.discriminator_ndf = 64
cfg.model.discriminator_use_sigmoid = False
```
4 layers ≈ 142×142 receptive field; 6 layers ≈ 574×574 (best for 2048²).

### Freezing Backbones
```python
cfg.model.freeze_backbone = True
cfg.model.freeze_bn = True
```

---

## Training Features

### Device Selection
```python
cfg.training.device = 'auto'  # CUDA → MPS → CPU fallback
```
`--device` CLI flag overrides config and errors if unavailable.

### Mixed Precision & Stability
```python
cfg.training.use_amp = True
cfg.training.grad_clip_norm = 1.0
```

### GAN Schedule
```python
cfg.training.gan_start_epoch = 5
cfg.training.d_steps_per_g_step = 1
```

### Learning-Rate Scheduling
```python
cfg.optimizer.scheduler = 'cosine'
cfg.optimizer.scheduler_min_lr = 1e-6
cfg.optimizer.scheduler_warmup_epochs = 5
```
Other options: `step`, `plateau`, `none`.

### Checkpointing & Logging
```python
cfg.training.checkpoint_dir = './checkpoints'
cfg.training.save_every_n_epochs = 5
cfg.training.save_best_only = False

cfg.training.use_tensorboard = True
cfg.training.log_every_n_steps = 10
cfg.training.log_images_every_n_epochs = 5
cfg.training.use_wandb = False
```

You can override the checkpoint directory at runtime with the CLI flag `--checkpoint-dir`:
```bash
python train.py --input-dir ./data/input --output-dir ./data/output --checkpoint-dir ./my_checkpoints
```

---

## Run Inference

`training/run_inference.py` previews a trained checkpoint using either dataset samples or custom PNG renders.

```bash
# Dataset sample (default)
python training/run_inference.py \
  --checkpoint checkpoints/latest.pth \
  --sample-idx 12 \
  --out-dir inference_outputs/sample_12

# Custom renders (directory with exactly three PNGs)
python training/run_inference.py \
  --checkpoint checkpoints/latest.pth \
  --input-dir ./my_renders \
  --out-dir inference_outputs/custom_material
```

Arguments:
- `--checkpoint` (required): generator checkpoint
- `--sample-idx`: dataset sample index if `--input-dir` omitted (default `0`)
- `--input-dir`: directory with three PNG renders (sorted alphabetically)
- `--out-dir`: folder for predicted maps plus denormalized input views

The script mirrors the saved config’s resize/normalize transforms so custom renders match training preprocessing.

---

## Monitoring & Logging

### Console
```
Epoch 10: 100%|██████████| 1000/1000 [12:34<00:00, 1.32it/s, G_loss=0.123, D_loss=0.568]
Epoch 10 Summary:
  Generator Loss: 0.123
  Discriminator Loss: 0.568
[Validation] Epoch 10 - Loss: 0.110
Saved checkpoint: checkpoints/checkpoint_epoch_0010.pth
Saved best model: checkpoints/best_model.pth
```

### TensorBoard Signals
- Training: generator/discriminator losses, per-map L1/MAE, SSIM, normal angle, LR
- Validation: overall + per-map PSNR/SSIM/MAE, normal angle statistics, comparison grids, input views
- Images: predicted vs ground truth PBR maps, error heatmaps, denormalized renders every `log_images_every_n_epochs`

Launch viewer:
```bash
tensorboard --logdir checkpoints/logs
```

---

## Troubleshooting

| Issue | Mitigation |
| --- | --- |
| Only CPU detected | `python -c "import torch; print(torch.cuda.is_available())"`; reinstall CUDA build or pass `--device cuda` to see explicit error |
| OOM | Lower `batch_size`, switch to `resnet18`, reduce transformer depth, keep AMP enabled |
| OOM (Fragmentation) | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable before running scripts |
| Unstable GAN | Lower `g_lr/d_lr`, tighter `grad_clip_norm`, delay GAN start, freeze BN |
| Slow convergence | Increase LR, use cosine scheduler with warmup, raise L1/normal weights |
| Poor normal quality | Boost `w_normal` & `w_normal_map`, verify dataset normalization |

---

## Advanced Usage

### Custom Losses
Add modules in `losses/losses.py` and integrate into `HybridLoss`.

### Export for Inference
```python
ckpt = torch.load('checkpoints/best_model.pth')
state = ckpt['generator_state_dict']
cfg = ckpt['config']
torch.save({'config': cfg, 'model_state_dict': state}, 'model_inference.pth')
```

### Inline Inference Example
```python
from train import MultiViewPBRGenerator
ckpt = torch.load('model_inference.pth')
model = MultiViewPBRGenerator(ckpt['config']).eval().cuda()
model.load_state_dict(ckpt['model_state_dict'])
with torch.no_grad():
    outputs = model(inputs)
```

---

## Student Training (Distillation)

For mobile deployment (Core ML), we train a lightweight "Student" model (MobileNetV3 + Small ViT) to mimic the heavy "Teacher" model (ResNet + Large ViT).

### Training Configurations

| Config | Input Size | Output Size | Use Case |
|--------|-----------|-------------|----------|
| `mobilenetv3_512.py` | 512×512 | 1024×1024 (SR 2×) | **Recommended for iPhone** (ViT bottleneck) |
| `convattn_student.py` | 1024×1024 | 1024×1024 (No SR) | **Alternative for iPhone** (PLK bottleneck, higher resolution potential) |
| `mobilenetv3_2048.py` | 2048×2048 | 2048×2048 | Desktop/high-memory devices |

#### Architecture Comparison

| Architecture | Bottleneck | Memory Scaling | Max ANE Resolution | Notes |
|--------------|------------|----------------|-------------------|-------|
| MobileNetV3 + ViT | Vision Transformer | O(N²) | ~512→1024 | Proven quality, attention-based |
| MobileNetV3 + ConvAttn | PLK (Pre-computed Large Kernel) | O(N) | ~1024→1024 (estimated) | Based on ESC paper, linear memory |

The **ConvAttn** architecture uses Pre-computed Large Kernels (PLK) instead of Vision Transformer attention. This trades some representational power for O(N) memory scaling, enabling higher resolutions on Apple Neural Engine.

### 1. Generate Distillation Shards
Instead of running the heavy teacher model during training (which is slow and VRAM-heavy), we pre-compute the teacher's outputs and save them as "shards" (.pt files).

**Note:** Shards now only store the teacher's predictions (in float16) to save disk space. The original inputs and targets are loaded from the PNG dataset on-the-fly during student training.

**For 512×512 student training, generate shards at 1024×1024** (matching the student's SR output):

```bash
# Generate 1024×1024 shards from a trained teacher (recommended)
python teacher_infer.py \
  --checkpoint checkpoints/best_model.pth \
  --data-root ./data \
  --shards-dir ./data/shards_1024 \
  --shard-output-size 1024
```

**For 2048×2048 student training:**

```bash
# Generate 2048×2048 shards (full resolution)
python teacher_infer.py \
  --checkpoint checkpoints/best_model.pth \
  --data-root ./data \
  --shards-dir ./data/shards_2048
```

*   `--shard-output-size 1024`: Downsamples teacher outputs to 1024×1024 before saving (75% smaller files).
*   `--shard-size 8`: Keeps shard files manageable.

### 2. Train Student Model
Train the student model using the pre-computed shards. This is much faster and uses less VRAM.

#### Option A: ViT Student (Recommended)

**For iPhone deployment (512 input → 1024 output):**

```bash
python student/train.py \
  --config configs/mobilenetv3_512.py \
  --shards-dir ./data/shards_1024 \
  --input-dir ./data/input \
  --output-dir ./data/output \
  --checkpoint-dir ./checkpoints_student
```

#### Option B: ConvAttn Student (Experimental)

Uses PLK (Pre-computed Large Kernel) bottleneck instead of ViT attention. May enable higher resolutions on ANE due to O(N) memory scaling.

**For iPhone deployment (512 input → 1024 output via 4× SR):**

```bash
python student/train.py \
  --config configs/convattn_student.py \
  --shards-dir ./data/shards_1024 \
  --input-dir ./data/input \
  --output-dir ./data/output \
  --checkpoint-dir ./checkpoints_convattn
```

**Training Notes:**
- ConvAttn typically converges in **50-70 epochs** (config default: 60)
- Uses the same distillation shards as ViT student
- The PLK kernel size is 17×17 with 3 blocks

#### Option C: High-Resolution (Desktop)

**For high-resolution training (2048 input → 2048 output):**

```bash
python student/train.py \
  --config configs/mobilenetv3_2048.py \
  --shards-dir ./data/shards_2048 \
  --input-dir ./data/input \
  --output-dir ./data/output \
  --checkpoint-dir ./checkpoints_student
```

*   `--input-dir` and `--output-dir` are **required** because the student loader reads the original PNGs from disk while fetching teacher predictions from the shards.
*   The dataset automatically resizes inputs to `image_size` and targets to `output_size` from the config.

### 3. Core ML Optimization
The `configs/mobilenetv3_512.py` configuration is specifically tuned for Apple Neural Engine (ANE):
*   **Encoder**: MobileNetV3-Large with lightweight transformer fusion.
*   **Transformer**: Reduced dimension (256) and depth (2) to fit memory bandwidth constraints.
*   **Input Resolution**: 512×512 model inputs for memory efficiency.
*   **Output Resolution**: 1024×1024 via trained SR head (2× upscale).
*   **SR Head**: Trained neural upscaler from 512 to 1024 (preserved, not stripped).
*   **Final Output**: 2048×2048 via Lanczos upscaling on-device.
*   **Output Format**: Lossless PNG files (preferred for artist/texture workflows).

This pipeline produces a model capable of running on iPhone 12 and newer (iOS 17+ required).

### 4. Convert to Core ML
Once the student model is trained, convert it to a `.mlpackage` for iOS deployment using the provided converter script.

**Note:** A pre-compiled model is already included in the repository at `mobile_app/ios/pbr_model.mlpackage`. Use this script only if you have trained a new student model and want to deploy it.

```bash
# Run on macOS (requires coremltools)
python3 training/coreml/converter.py \
  checkpoints/best_student.pth \
  --output mobile_app/ios/pbr_model.mlpackage
```

This script automatically:
1. Detects the student architecture.
2. Keeps the trained Super-Resolution head (512→1024 neural upscaling).
3. Configures 512×512 input for optimal iPhone memory usage.
4. Applies ImageNet normalization directly into the Core ML graph.
5. Applies constant elimination and dead code elimination.
6. Exports with FP16 precision (palettization disabled by default for quality).
7. Sets minimum deployment target to iOS 17 for best ANE compatibility.

On-device, outputs are further upscaled from 1024 to 2048 using Lanczos resampling.

**Converter Flags:**
- `--no-fp16`: Disable FP16 quantization (use if you see artifacts).
- `--palettization`: Enable 8-bit weight clustering (smaller model, may reduce quality).
- `--test-resolution <int>`: Convert at a custom resolution (e.g., 768) for ANE memory testing. By default, this bypasses the SR head so output equals input resolution.
- `--use-sr`: When used with `--test-resolution`, keeps the trained SR head active. Output resolution = input × SR scale (e.g., `--test-resolution 384 --use-sr` → 384×2 = 768 output).

**Example: Test ANE Memory Limits**
```bash
# Test 768×768 without SR (validates ANE memory budget)
python3 training/coreml/converter.py checkpoints/best_student.pth \
  --test-resolution 768 --output pbr_768_test.mlpackage

# Test 512×512 input with SR head (512→1024 output)
python3 training/coreml/converter.py checkpoints/best_student.pth \
  --test-resolution 512 --use-sr --output pbr_512_sr.mlpackage
```

### 5. Run Core ML Inference (CLI)
You can test the compiled `.mlpackage` on macOS using the provided inference script. This is useful for verifying the model's output without deploying to a device.

```bash
python3 training/coreml/run_inference.py \
  mobile_app/ios/pbr_model.mlpackage \
  --input path/to/folder_with_3_images \
  --output inference_results
```

*   **Input**: A folder containing at least 3 images (png/jpg). The script will use the first 3 sorted alphabetically.
*   **Output**: Saves `albedo.png`, `normal.png`, `roughness.png`, and `metallic.png` to the specified output directory.
*   **Note**: The script resizes inputs to 512×512, the model upscales to 1024×1024 via the trained SR head, then outputs are upscaled to 2048×2048 via Lanczos.
