# NeuroPBR Training

Multi-view fusion GAN training for PBR texture reconstruction from rendered images.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default config
python train.py --data-root /path/to/your/data

# Train with dirty renders instead of clean
python train.py --data-root /path/to/your/data --use-dirty

# Train with explicit directories (bypass data-root layout)
python train.py --input-dir /path/to/input --output-dir /path/to/output --metadata-path /path/to/render_metadata.json

# Quick test (small model, 10 epochs)
python train.py --config quick_test --data-root /path/to/your/data

# Monitor training
tensorboard --logdir checkpoints/logs
```

## Documentation

📖 **[Complete Training Guide](TRAINING_GUIDE.md)** - Read this first!

The comprehensive guide covers:
- Dataset setup and structure
- Configuration system (loss weights, model options)
- Command reference and workflows
- Troubleshooting and optimization
- Advanced usage and implementation details

## Features

✅ Multi-view fusion with Vision Transformer  
✅ Flexible loss system (L1 + SSIM + Normal + GAN + Perceptual)  
✅ Mixed precision training (AMP)  
✅ Automatic train/val split  
✅ Checkpointing and resume  
✅ TensorBoard logging  
✅ Multiple preset configs

## File Structure

```
training/
├── train.py              # Main training script
├── train_config.py       # Configuration system
├── TRAINING_GUIDE.md     # 📖 Complete documentation
├── requirements.txt      # Dependencies
├── configs/              # Example configurations
│   ├── high_quality.py
│   ├── fast_iteration.py
│   └── normal_focused.py
├── models/               # Model architectures
│   ├── encoders/
│   ├── decoders/
│   └── transformers/
├── losses/               # Loss functions
├── utils/                # Dataset and utilities
└── Test/                 # Unit tests
```

## Requirements

- Python 3.8+
- PyTorch 2.9.0
- CUDA-capable GPU (8GB+ VRAM recommended)
- See `requirements.txt` for full list

## Dataset Structure

```
your_data/
├── input/
│   ├── clean/sample_XXXX/{0,1,2}.png        (default training input)
│   ├── dirty/sample_XXXX/{0,1,2}.png        (optional, used with --use-dirty)
│   └── render_metadata.json                 (sample → material mapping)
└── output/
    └── material_name/
        ├── albedo.png                       (ground truth PBR maps)
        ├── roughness.png
        ├── metallic.png
        └── normal.png
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md#dataset-setup) for detailed setup instructions.

Prefer separate folders? Pass them directly via `--input-dir`, `--output-dir`, and (optionally) `--metadata-path` to bypass the default `{data_root}/input|output` layout. Clean renders remain the default input unless you pass `--use-dirty` or set `config.data.use_dirty_renders = True`.

## Common Commands

```bash
# Default training (ResNet50 + GAN)
python train.py --data-root ./data

# Train using dirty renders
python train.py --data-root ./data --use-dirty

# Explicit directories (input + GT)
python train.py --input-dir ./data/input --output-dir ./data/output

# No GAN (faster baseline)
python train.py --config lightweight --data-root ./data

# High quality (ResNet101 + perceptual)
python train.py --config configs/high_quality.py --data-root ./data

# Custom batch size
python train.py --data-root ./data --batch-size 8 --epochs 100

# Resume training
python train.py --resume checkpoints/latest.pth
```

## Configuration Examples

**Emphasize normal quality:**
```python
config.loss.w_normal = 1.0
config.loss.w_normal_map = 2.0
```

**Enable perceptual loss:**
```python
config.loss.use_perceptual = True
config.loss.w_perceptual = 0.2
```

**Disable GAN:**
```python
config.model.use_gan = False
config.loss.w_gan = 0.0
```

## Support

- **Documentation:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Issues:** GitHub Issues
- **Tests:** Run `pytest` in Test/ directory

---
