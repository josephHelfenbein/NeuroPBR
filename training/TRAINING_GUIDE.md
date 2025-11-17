# NeuroPBR Training Guide

**Complete guide for training multi-view fusion GAN models to reconstruct PBR textures from rendered images.**

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Dataset Setup](#dataset-setup)
4. [Configuration System](#configuration-system)
5. [Loss Configuration](#loss-configuration)
6. [Model Architecture](#model-architecture)
7. [Training Features](#training-features)
8. [Command Reference](#command-reference)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Usage](#advanced-usage)
12. [File Structure](#file-structure)

---

## Overview

NeuroPBR trains a deep learning model that:

1. Takes **3 rendered views** (with artifacts like smudges, scratches)
2. Uses **multi-view fusion** with Vision Transformer
3. Outputs **4 PBR maps** (albedo, roughness, metallic, normal) at 2048×2048
4. Trained with **GAN + reconstruction losses**

### Architecture

```
Input: 3 Rendered Views (dirty) → (B, 3, 3, 2048, 2048)
  ↓
Shared Encoder (ResNet/UNet) → 3 Latent Representations @ 64×64
  ↓
ViT Cross-View Fusion (attention) → Fused Latent (B, 2048, 64, 64)
  ↓
Multi-Head Decoder + Super-Resolution (2×) → 2048×2048
  ↓
Output: 4 PBR Maps
  ├── albedo (3ch, sigmoid)
  ├── roughness (1ch, sigmoid)
  ├── metallic (1ch, sigmoid)
  └── normal (3ch, normalized)

Discriminator: 
  - Simple: 4-layer PatchGAN (~142×142 receptive field)
  - Configurable: 6-layer PatchGAN (~574×574 receptive field, recommended)
```

### Key Features

✅ **Multi-view fusion** - Leverages all 3 views with ViT attention  
✅ **Flexible losses** - Switch between loss combinations via config  
✅ **GAN training** - Optional adversarial training for realism  
✅ **Mixed precision** - Faster training, less memory  
✅ **Auto train/val split** - Automatic 90/10 split  
✅ **Resume training** - Save/load checkpoints  
✅ **TensorBoard logging** - Comprehensive metrics  
✅ **Production-ready** - Gradient clipping, LR scheduling, validation

## Quick Start

### 1. Install Dependencies
```bash
cd training
pip install -r requirements.txt
```

**Optional dependencies** (uncomment in requirements.txt if needed):
- `wandb` - For Weights & Biases logging
- `opencv-python` - For advanced visualization
- `matplotlib` - For plotting

### 2. Verify Dataset Structure

**Required structure:**
```
data/
├── input/
│   ├── clean/
│   │   ├── sample_0000/
│   │   │   ├── 0.png  # Clean render view 1
│   │   │   ├── 1.png  # Clean render view 2
│   │   │   └── 2.png  # Clean render view 3
│   │   ├── sample_0001/
│   │   └── ...
│   ├── dirty/
│   │   ├── sample_0000/
│   │   │   ├── 0.png  # Dirty render view 1 (used for training)
│   │   │   ├── 1.png  # Dirty render view 2
│   │   │   └── 2.png  # Dirty render view 3
│   │   ├── sample_0001/
│   │   └── ...
│   └── render_metadata.json  # Maps sample_XXXX -> material_name
└── output/
    ├── material_0/
    │   ├── albedo.png      # Ground truth PBR maps
    │   ├── roughness.png
    │   ├── metallic.png
    │   └── normal.png
    ├── material_1/
    └── ...
```

The `render_metadata.json` should look like:
```json
{
  "sample_0000": "material_0",
  "sample_0001": "material_1",
  ...
}
```

### 3. Train with Default Config
```bash
python train.py --data-root ./data
```

### 4. Train with Custom Settings
```bash
# Quick test (small model, few epochs)
python train.py --config quick_test --data-root ./data --batch-size 2

# Without GAN (faster, simpler)
python train.py --config lightweight --data-root ./data

# Custom config
python train.py --config configs/my_config.py --data-root ./data
```

### 5. Resume Training
```bash
python train.py --resume checkpoints/checkpoint_epoch_0050.pth
```

---

## Dataset Setup

### Understanding the Dataset Structure

The dataset separates **input renders** (what the model sees) from **output PBR maps** (what it should predict).

**Key concepts:**
- **Input renders**: 3 views of rendered materials (can be clean or dirty)
- **Output PBR maps**: Ground truth material properties (albedo, roughness, metallic, normal)
- **Metadata mapping**: Links render samples to material names via JSON

### Directory Structure

```
your_data/
├── input/
│   ├── clean/                    # Required: clean renders (default input)
│   │   ├── sample_0000/
│   │   │   ├── 0.png            # View 1 (clean)
│   │   │   ├── 1.png            # View 2 (clean)
│   │   │   └── 2.png            # View 3 (clean)
│   │   ├── sample_0001/
│   │   └── ...
│   ├── dirty/                    # Optional: dirty renders (with artifacts)
│   │   ├── sample_0000/
│   │   │   ├── 0.png            # View 1 (dirty - training input)
│   │   │   ├── 1.png            # View 2 (dirty)
│   │   │   └── 2.png            # View 3 (dirty)
│   │   ├── sample_0001/
│   │   └── ...
│   └── render_metadata.json      # Required: sample → material mapping
└── output/                        # Required: ground truth PBR maps
    ├── wood_oak/
    │   ├── albedo.png            # RGB albedo map
    │   ├── roughness.png         # Grayscale roughness (R channel used)
    │   ├── metallic.png          # Grayscale metallic (R channel used)
    │   └── normal.png            # RGB normal map (tangent space)
    ├── metal_rusty/
    └── ...
```

### Creating render_metadata.json

This file maps sample folder names to material names:

```json
{
  "sample_0000": "wood_oak",
  "sample_0001": "metal_rusty",
  "sample_0002": "concrete_rough",
  "sample_0003": "plastic_glossy"
}
```

**Location:** `{data_root}/input/render_metadata.json`

**Purpose:** Connects the 3 rendered views in `input/clean/sample_XXXX/` (or `input/dirty/...` when enabled) to the ground truth PBR maps in `output/material_name/`.

### Training on Clean vs Dirty Renders

**Default (recommended):** Train on **clean renders**
```python
config.data.use_dirty_renders = False  # Default
```
Uses: `input/clean/sample_XXXX/{0,1,2}.png`

**Artifact robustness:** Train on **dirty renders**
```python
config.data.use_dirty_renders = True
```
Uses: `input/dirty/sample_XXXX/{0,1,2}.png`

### Train/Val Split

The dataset **automatically splits** into train and validation:

```python
config.data.val_ratio = 0.1  # 10% validation (default)
config.training.seed = 42     # Seed for reproducible shuffling
```

- **Shuffled:** Samples are randomly shuffled before splitting (not alphabetical)
- **Reproducible:** Same seed = same split every run
- **Representative:** Validation set covers diverse samples, not just first N%
- **No overlap:** Train samples don't appear in validation
- **Configurable:** Change `val_ratio` to adjust split (0.15 = 15% val, etc.)

### Verifying Your Dataset

```bash
# Check structure
ls your_data/input/clean/sample_*
ls your_data/output/*/
cat your_data/input/render_metadata.json

# Count samples
echo "Clean renders:" (ls your_data/input/clean/ | wc -l)
echo "Output materials:" (ls your_data/output/ | wc -l)

# Test loading
cd training
python -c "
from utils.dataset import PBRDataset
ds = PBRDataset(
  input_dir='../your_data/input',
  output_dir='../your_data/output',
  metadata_path='../your_data/input/render_metadata.json',
  transform_mean=[0.5, 0.5, 0.5],
  transform_std=[0.5, 0.5, 0.5],
  use_dirty=True
)
print(f'Total samples: {len(ds)}')
inputs, outputs = ds[0]
print(f'Input shape: {inputs.shape}')   # (3, 3, H, W)
print(f'Output shape: {outputs.shape}')  # (4, 3, H, W)
print('✓ Dataset loaded successfully!')
"
```

### Common Dataset Issues

**Issue:** `FileNotFoundError: render_metadata.json`  
**Fix:** Create the JSON file at `{data_root}/input/render_metadata.json`

**Issue:** `KeyError: sample_XXXX not in metadata`  
**Fix:** Add missing sample mappings to `render_metadata.json`

**Issue:** `No samples found`  
**Fix:** Check that:
- `input/dirty/sample_XXXX/` folders exist with {0,1,2}.png
- `output/material_name/` folders exist with all 4 PBR maps
- Material names in metadata match folder names in `output/`

---

## Configuration System

### Available Preset Configs

1. **`default`** - Full multi-view fusion + GAN
   - ResNet50 encoder
   - 4-layer ViT fusion
   - GAN training starts at epoch 5
   - Best quality, slower training

2. **`quick_test`** - Fast testing
   - ResNet18 encoder
   - 2-layer ViT fusion
   - 10 epochs
   - Good for debugging

3. **`lightweight`** - No GAN
   - Same architecture but no adversarial training
   - Faster convergence
   - Good baseline

### Create Custom Config

```python
# configs/my_config.py
from train_config import TrainConfig

def get_config():
    config = TrainConfig()
    
    # Model settings
    config.model.encoder_backbone = "resnet101"  # Larger encoder
    config.model.transformer_depth = 6  # Deeper transformer
    config.model.use_gan = True
    
    # Training settings
    config.training.epochs = 200
    config.training.batch_size = 8
    config.data.batch_size = 8
    
    # Loss weights
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.5
    config.loss.w_normal = 0.7
    config.loss.w_gan = 0.1
    config.loss.w_perceptual = 0.1  # Enable perceptual loss
    config.loss.use_perceptual = True
    
    # Optimizer
    config.optimizer.g_lr = 1e-4
    config.optimizer.d_lr = 4e-4
    config.optimizer.scheduler = "cosine"
    
    return config
```

Then run:
```bash
python train.py --config configs/my_config.py --data-root ./data
```

## Loss Configuration

The training script supports flexible loss combinations through the `LossConfig`:

### Available Loss Components

| Loss | Description | Config Key | Default Weight |
|------|-------------|------------|----------------|
| **L1** | Per-pixel reconstruction | `w_l1` | 1.0 |
| **SSIM** | Structural similarity (albedo) | `w_ssim` | 0.3 |
| **Normal** | Angular consistency (normals) | `w_normal` | 0.5 |
| **GAN** | Adversarial loss | `w_gan` | 0.05 |
| **Perceptual** | VGG feature matching | `w_perceptual` | 0.0 (disabled) |

### Per-Map L1 Weights

```python
config.loss.w_albedo = 1.0      # Albedo reconstruction
config.loss.w_roughness = 1.0   # Roughness reconstruction  
config.loss.w_metallic = 1.0    # Metallic reconstruction
config.loss.w_normal_map = 1.0  # Normal map reconstruction
```

### Example: Emphasize Normal Quality

```python
config.loss.w_normal = 1.0       # Higher angular loss
config.loss.w_normal_map = 2.0   # Higher L1 for normals
config.loss.w_albedo = 0.8       # Slightly lower albedo weight
```

### Example: Photography-Quality Albedo

```python
config.loss.w_albedo = 1.5       # Higher albedo weight
config.loss.w_ssim = 0.7         # Strong structural similarity
config.loss.w_perceptual = 0.2   # Add perceptual loss
config.loss.use_perceptual = True
```

### GAN Loss Types

```python
# Hinge loss (default, more stable)
config.loss.gan_loss_type = "hinge"

# Binary cross-entropy (classic GAN)
config.loss.gan_loss_type = "bce"
```

### Disable GAN Training

```python
config.model.use_gan = False
config.loss.w_gan = 0.0
```

## Model Architecture Options

### Encoder Types

```python
# ResNet encoder (pretrained, best quality)
config.model.encoder_type = "resnet"
config.model.encoder_backbone = "resnet50"  # 18, 34, 50, 101, 152
config.model.encoder_stride = 1  # 1 for 2048→2048, 2 for 1024→2048

# Custom UNet encoder (from scratch)
config.model.encoder_type = "unet"
config.model.encoder_channels = [64, 128, 256, 512, 1024, 2048]

# Strided UNet encoder (no max pooling)
config.model.encoder_type = "unet_stride"
```

### Transformer Settings

```python
# Standard settings
config.model.transformer_dim = 2048
config.model.transformer_num_heads = 32
config.model.transformer_depth = 4

# Lighter (faster)
config.model.transformer_num_heads = 16
config.model.transformer_depth = 2

# Heavier (more capacity)
config.model.transformer_num_heads = 32
config.model.transformer_depth = 6
config.model.transformer_mlp_ratio = 4
```

### Discriminator Options

```python
# Configurable discriminator (recommended for 2048×2048)
config.model.discriminator_type = "configurable"
config.model.discriminator_n_layers = 6  # 4-6 layers (more = larger receptive field)
config.model.discriminator_ndf = 64      # Base filters
config.model.discriminator_use_sigmoid = False  # False for hinge, True for BCE

# Simple discriminator (original 4-layer)
config.model.discriminator_type = "simple"
config.model.discriminator_ndf = 64
```

**Receptive field by layer count (2048×2048 input):**
- 4 layers: ~142×142 pixels (6.9% coverage)
- 5 layers: ~286×286 pixels (14% coverage)
- 6 layers: ~574×574 pixels (28% coverage) ← Recommended for 2048×2048

### Freeze Pretrained Weights

```python
# Freeze early ResNet layers (fine-tune only later layers)
config.model.freeze_backbone = True

# Freeze BatchNorm (stable training)
config.model.freeze_bn = True
```

## Training Features

### Mixed Precision Training
Automatically enabled by default for faster training:
```python
config.training.use_amp = True  # Automatic Mixed Precision
```

### Gradient Clipping
Prevents gradient explosion:
```python
config.training.grad_clip_norm = 1.0  # Max gradient norm
```

### GAN Training Schedule
Start GAN training after model learns basics:
```python
config.training.gan_start_epoch = 5  # Warmup epochs before GAN
config.training.d_steps_per_g_step = 1  # D updates per G update
```

### Learning Rate Scheduling

```python
# Cosine annealing (smooth decay) - Recommended
config.optimizer.scheduler = "cosine"
config.optimizer.scheduler_min_lr = 1e-6
config.optimizer.scheduler_warmup_epochs = 5  # Linear warmup from 10% to 100% of lr

# Step decay (periodic drops)
config.optimizer.scheduler = "step"
config.optimizer.step_size = 30  # Drop every 30 epochs
config.optimizer.gamma = 0.1     # Multiply by 0.1

# Adaptive (ReduceLROnPlateau)
config.optimizer.scheduler = "plateau"
config.optimizer.patience = 10
config.optimizer.factor = 0.5

# No scheduling
config.optimizer.scheduler = "none"
```

**Warmup:** If `scheduler_warmup_epochs > 0`, learning rate starts at 10% of target and linearly increases over N epochs before the main scheduler takes over.

### Checkpointing

```python
config.training.checkpoint_dir = "./checkpoints"
config.training.save_every_n_epochs = 5  # Save every 5 epochs
config.training.save_best_only = False   # If True, only saves best + latest (saves disk space)
```

Checkpoints saved:
- `checkpoint_epoch_XXXX.pth` - Regular checkpoints (if save_best_only=False)
- `best_model.pth` - Best validation loss
- `latest.pth` - Most recent checkpoint (always saved)

### Logging

```python
# TensorBoard (default)
config.training.use_tensorboard = True
config.training.log_every_n_steps = 10
config.training.log_images_every_n_epochs = 5  # Log visualizations every 5 epochs

# WandB (optional - requires: pip install wandb)
config.training.use_wandb = True
config.training.wandb_project = "neuropbr"
config.training.wandb_run_name = "experiment_01"

# View logs
tensorboard --logdir checkpoints/logs
```

**What Gets Logged:**
- Training: losses, learning rates, per-map metrics
- Validation: loss, PSNR, SSIM, angular error, per-map metrics
- Images: input views, pred/target comparisons, error heatmaps (every N epochs)

## Multi-GPU Training (Coming Soon)

```bash
# 4 GPUs
torchrun --nproc_per_node=4 train.py --distributed --data-root ./data

# 2 nodes, 4 GPUs each
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="192.168.1.1" \
  --master_port=29500 \
  train.py --distributed --data-root ./data
```

---

## Command Reference

### Basic Commands

```bash
# Default config (ResNet50 + ViT + GAN, 100 epochs)
python train.py --data-root /path/to/data

# Quick test (ResNet18, 10 epochs)
python train.py --config quick_test --data-root /path/to/data

# No GAN (faster, simpler baseline)
python train.py --config lightweight --data-root /path/to/data

# High quality (ResNet101 + perceptual loss)
python train.py --config configs/high_quality.py --data-root /path/to/data

# Custom batch size and epochs
python train.py --data-root /path/to/data --batch-size 8 --epochs 100

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_0050.pth

# Custom checkpoint directory
python train.py --data-root /path/to/data --checkpoint-dir ./my_experiment
```

### Config Presets Quick Reference

| Preset | Model | Depth | Epochs | GAN | Use Case |
|--------|-------|-------|--------|-----|----------|
| `default` | ResNet50 | 4 | 100 | ✅ | Production quality |
| `quick_test` | ResNet18 | 2 | 10 | ✅ | Fast debugging |
| `lightweight` | ResNet50 | 4 | 100 | ❌ | Baseline (no GAN) |
| `configs/high_quality.py` | ResNet101 | 6 | 200 | ✅ | Best results |
| `configs/fast_iteration.py` | ResNet18 | 2 | 50 | ✅ | Quick experiments |
| `configs/normal_focused.py` | ResNet50 | 4 | 150 | ✅ | High-quality normals |

### Example Workflows

**Workflow 1: First-time training**
```bash
# Start with quick test to verify everything works
python train.py --config quick_test --data-root ./data --batch-size 2

# Monitor in separate terminal
tensorboard --logdir checkpoints/logs

# If successful, run full training
python train.py --data-root ./data --checkpoint-dir ./experiments/run_001
```

**Workflow 2: Hyperparameter tuning**
```bash
# Create custom config
cp configs/high_quality.py configs/my_experiment.py
# Edit configs/my_experiment.py with your settings

# Train
python train.py --config configs/my_experiment.py --data-root ./data
```

**Workflow 3: Resume interrupted training**
```bash
# Training was interrupted at epoch 47
python train.py --resume checkpoints/latest.pth

# Or resume from specific checkpoint
python train.py --resume checkpoints/checkpoint_epoch_0050.pth
```

---

## Monitoring & Logging

### Monitoring Training

### Console Output
```
Epoch 10: 100%|████████| 1000/1000 [12:34<00:00, 1.32it/s, G_loss=0.1234, D_loss=0.5678]

Epoch 10 Summary:
  Generator Loss: 0.1234
  Discriminator Loss: 0.5678

[Validation] Epoch 10 - Loss: 0.1100
Saved checkpoint: checkpoints/checkpoint_epoch_0010.pth
Saved best model: checkpoints/best_model.pth
```

### TensorBoard Metrics

**Training:**
- `train/g_loss` - Generator total loss
- `train/d_loss` - Discriminator loss
- `train/g_lr` - Current learning rate
- `train/loss_l1_total` - L1 reconstruction loss
- `train/loss_ssim` - SSIM loss
- `train/loss_normal` - Normal consistency loss
- `train/loss_gan_g` - Generator adversarial loss
- `train/l1_albedo`, `l1_roughness`, etc. - Per-map L1 losses
- `train/mae_albedo`, `train/rmse_albedo` - Metrics
- `train/normal_angle_deg` - Average normal angle error

**Validation:**
- `val/loss` - Validation loss
- `val/overall_psnr`, `val/overall_ssim` - Overall quality metrics
- `val/albedo_psnr`, `val/albedo_ssim`, `val/albedo_mae`, etc. - Per-map metrics
- `val/normal_angle_mean`, `val/normal_angle_median` - Normal error statistics
- `val/comparison` - Pred vs target comparison grid (images)
- `val/input_views` - Input render views (images)

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
config.data.batch_size = 2

# Smaller model
config.model.encoder_backbone = "resnet18"
config.model.transformer_depth = 2

# Disable AMP (uses more memory but sometimes helps)
config.training.use_amp = False
```

### Unstable Training
```python
# Lower learning rates
config.optimizer.g_lr = 1e-4
config.optimizer.d_lr = 2e-4

# More gradient clipping
config.training.grad_clip_norm = 0.5

# Start GAN later
config.training.gan_start_epoch = 10

# Freeze BatchNorm
config.model.freeze_bn = True
```

### Slow Convergence
```python
# Higher learning rate
config.optimizer.g_lr = 5e-4

# Warmup scheduler
config.optimizer.scheduler = "cosine"
config.optimizer.scheduler_warmup_epochs = 5

# More aggressive loss weights
config.loss.w_l1 = 2.0
config.loss.w_normal = 1.0
```

### Poor Normal Quality
```python
# Emphasize normal losses
config.loss.w_normal = 1.0
config.loss.w_normal_map = 2.0

# Ensure normalization in data preprocessing
config.transform.mean = [0.5, 0.5, 0.5]
config.transform.std = [0.5, 0.5, 0.5]
```

## Advanced Usage

### Custom Loss Function

You can modify `losses/losses.py` to add custom losses:

```python
class MyCustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        # Your loss computation
        loss = ...
        return loss * self.weight
```

Then integrate into `HybridLoss` or use separately in `train.py`.

### Export Model for Inference

```python
# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pth")

# Extract generator only
generator_state = checkpoint["generator_state_dict"]

# Save for inference
torch.save({
    "model_state_dict": generator_state,
    "config": checkpoint["config"]
}, "model_inference.pth")
```

### Inference Example

```python
from train import MultiViewPBRGenerator
from train_config import TrainConfig

# Load config and model
checkpoint = torch.load("model_inference.pth")
config = checkpoint["config"]
model = MultiViewPBRGenerator(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.cuda()

# Inference
with torch.no_grad():
    views = load_three_views()  # (1, 3, 3, 1024, 1024)
    pbr_maps = model(views)
    # pbr_maps["albedo"], pbr_maps["roughness"], etc.
```

## File Structure

```
training/
├── train.py              # Main training script
├── train_config.py       # Configuration dataclasses
├── requirements.txt      # Dependencies
├── models/
│   ├── encoders/
│   │   └── unet.py      # Encoder architectures
│   ├── decoders/
│   │   └── unet.py      # Decoder architectures
│   └── transformers/
│       └── vision_transformer.py  # ViT fusion
├── losses/
│   └── losses.py        # Loss functions & discriminator
├── utils/
│   ├── dataset.py       # Dataset & dataloader
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Visualization tools
└── Test/                # Unit tests
```

## Dataset Configuration

### Using Clean vs Dirty Renders

By default, the model trains on **clean renders**:
```python
config.data.use_dirty_renders = False  # Default
```

To emphasize artifact robustness, train on dirty renders:
```python
config.data.use_dirty_renders = True
```

### Train/Val Split

The dataset automatically splits into train/val:
```python
config.data.val_ratio = 0.1  # 10% for validation (default)
```

The split is deterministic based on sample order, so it's consistent across runs.

### Creating render_metadata.json

The `render_metadata.json` file maps sample names to material names:

```json
{
  "sample_0000": "wood_oak_planks",
  "sample_0001": "metal_rusted_steel",
  "sample_0002": "concrete_rough",
  ...
}
```

This file should be placed at: `{data_root}/input/render_metadata.json`

---

## Implementation Details

### What's Implemented

✅ **Complete training pipeline**
- Multi-view fusion generator (encoder + ViT + decoder)
- PatchGAN discriminator
- Flexible loss system (L1 + SSIM + Normal + GAN + Perceptual)
- Mixed precision training (AMP)
- Gradient clipping
- Learning rate scheduling
- Automatic train/val split
- Checkpointing (regular + best + latest)
- TensorBoard logging
- Resume from checkpoint

✅ **Model architectures**
- 3 encoder variants: UNet, UNetStride, ResNet (18/34/50/101/152)
- ViT cross-view fusion (configurable depth/heads)
- Multi-head decoder with super-resolution (2× or 4×)
- PatchGAN discriminator (70×70 receptive field)

✅ **Loss functions**
- WeightedL1Loss (per-map weighting)
- SSIMLoss (11×11 Gaussian window)
- NormalConsistencyLoss (angular error)
- VGGPerceptualLoss (optional, VGG16 features)
- GAN losses (hinge or BCE)
- HybridLoss (combines all with configurable weights)

✅ **Dataset utilities**
- PBRDataset with metadata mapping
- Automatic shuffled train/val split (reproducible with seed)
- Clean/dirty render selection
- Data normalization

✅ **Metrics & Visualization**
- Comprehensive validation metrics (PSNR, SSIM, angular error, MAE, RMSE)
- TensorBoard image logging with comparison grids
- Error heatmaps and normal map visualizations
- Per-map and overall metrics

✅ **Logging & Monitoring**
- TensorBoard integration (losses, metrics, images)
- WandB support (optional, with graceful fallback)
- Learning rate scheduling with warmup
- Image logging every N epochs

### What's NOT Implemented (Future Work)

⚠️ **Multi-GPU training** - Structure ready, needs DDP implementation  
⚠️ **Data augmentation** - Only basic transforms (add random crops, color jitter)  
⚠️ **Inference script** - No standalone inference utility  
⚠️ **Export utilities** - No ONNX/TorchScript export  

### Files Created

```
training/
├── train.py                     # Main training script (670+ lines)
├── train_config.py              # Configuration system (290+ lines)
├── TRAINING_GUIDE.md            # This file
├── configs/
│   ├── high_quality.py         # Best quality preset
│   ├── fast_iteration.py       # Quick experiments
│   └── normal_focused.py       # Normal-focused training
└── requirements.txt             # Updated dependencies
```

### Design Decisions

**Why multi-view fusion?**
- Leverages all 3 rendered views
- Cross-attention captures view-consistent features
- More robust to per-view artifacts

**Why GAN training?**
- Produces sharper, more realistic details
- Helps with high-frequency textures
- Optional (can disable for faster baseline)

**Why separate input/output?**
- Clean separation of concerns
- Flexible mapping via metadata
- Easy to add new materials or renders

**Why automatic train/val split?**
- No manual data splitting needed
- Deterministic (same split every run)
- Configurable ratio

### Performance Tips

**Maximize throughput:**
```python
config.data.batch_size = 8  # Larger batches if GPU memory allows
config.data.num_workers = 8  # More workers for data loading
config.training.use_amp = True  # Mixed precision (faster)
config.data.pin_memory = True  # Faster GPU transfer
config.data.persistent_workers = True  # Keep workers alive
```

**Minimize memory:**
```python
config.data.batch_size = 2  # Smaller batches
config.model.encoder_backbone = "resnet18"  # Smaller model
config.model.transformer_depth = 2  # Shallower transformer
config.training.use_amp = True  # Mixed precision saves memory
```

**Best quality:**
```python
config.model.encoder_backbone = "resnet101"  # Large encoder
config.model.transformer_depth = 6  # Deep fusion
config.loss.use_perceptual = True  # Add perceptual loss
config.loss.w_perceptual = 0.2
config.training.epochs = 200  # More training
```

---

## FAQ

**Q: How much GPU memory do I need?**  
A: For 2048×2048 images: Minimum 16GB (batch_size=2 with ResNet50). Recommended 24GB+ for larger batches. For 1024×1024: 8GB works with batch_size=2.

**Q: How long does training take?**  
A: ~18-24 hours for 100 epochs with ResNet50 + 6-layer discriminator on 2048×2048 images (single V100/A100, depends on dataset size).

**Q: Can I train on CPU?**  
A: Technically yes, but very slow (not recommended). Use at least 1 GPU with 16GB+ VRAM.

**Q: Do I need clean renders?**  
A: No, `input/clean/` is optional. Only `input/dirty/` is required for training.

**Q: What resolution should my renders be?**  
A: Default is 2048×2048 (loaded and output at same resolution). Can be configured to 1024×1024 for faster training. Higher resolution = better quality but slower and more memory.

**Q: How do I create the metadata JSON?**  
A: Write a script to map your sample folders to material names, or create it manually if small dataset.

**Q: Can I use different numbers of views?**  
A: Currently hardcoded to 3 views. Would need to modify `ViTCrossViewFusion` and dataset for different numbers.

**Q: What if my PBR maps are different resolutions?**  
A: They're automatically resized. Make sure they're square (or adjust dataset transforms).

**Q: How do I know if training is working?**  
A: Watch TensorBoard metrics. Generator loss should decrease, normal angle error should drop below 20°, discriminator should stay balanced.

---

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/josephHelfenbein/NeuroPBR/issues
- Documentation: See this guide and inline code comments
- Tests: Run `pytest` in the `Test/` directory
