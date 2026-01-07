#!/usr/bin/env python3
"""
Quick validation script to check if metallic head is learning.
Run this after 1-2 epochs to verify the fix is working before committing to full training.

Usage:
    python quick_validate.py /path/to/checkpoint_epoch_0001.pth --input-dir /path/to/input --output-dir /path/to/output

What to look for:
  ✓ GOOD: metallic_std > 0.01 (model is outputting varied values)
  ✓ GOOD: metallic gradient > 1e-4 on metallic samples
  ✗ BAD:  metallic_std ≈ 0 (collapsed, abort and investigate)
  ✗ BAD:  metallic gradient < 1e-6 (not learning, abort)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Quick validation for metallic learning")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/ultra_stable.py")
    args = parser.parse_args()
    
    sys.path.insert(0, str(Path(__file__).parent))
    from train import MultiViewPBRGenerator, denormalize_target
    from train_config import TrainConfig
    from utils.dataset import PBRDataset
    from losses.losses import HybridLoss
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.get_config()
    else:
        config = TrainConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MultiViewPBRGenerator(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('generator_state_dict', ckpt.get('model_state_dict', {}))
    new_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.train()
    
    # Load dataset
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir) if args.output_dir else input_path.parent / "output"
    metadata_path = input_path / "render_metadata.json"
    
    dataset = PBRDataset(
        input_dir=str(input_path),
        output_dir=str(output_path),
        metadata_path=str(metadata_path),
        transform_mean=config.transform.mean,
        transform_std=config.transform.std,
        image_size=config.data.image_size,
        output_size=config.data.output_size,
        curriculum_mode=0
    )
    
    # Build loss
    loss_config = {
        "w_l1": config.loss.w_l1,
        "w_ssim": config.loss.w_ssim,
        "w_normal": config.loss.w_normal,
        "w_gan": 0.0,
        "w_albedo": config.loss.w_albedo,
        "w_roughness": config.loss.w_roughness,
        "w_metallic": config.loss.w_metallic,
        "w_normal_map": config.loss.w_normal_map,
        "gan_loss_type": "hinge",
        "metallic_boost": getattr(config.loss, 'metallic_boost', 10.0)
    }
    criterion = HybridLoss(loss_config).to(device)
    
    epoch = ckpt.get('epoch', '?')
    print(f"\n{'='*60}")
    print(f"QUICK VALIDATION - Epoch {epoch}")
    print(f"{'='*60}")
    
    # Test on 10 random samples
    import random
    indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
    # Track stats for all heads
    stats = {
        'metallic': {'std': [], 'grad': []},
        'roughness': {'std': [], 'grad': []},
        'normal': {'z': [], 'grad': []},
        'albedo': {'std': [], 'grad': []}
    }
    
    for i in indices:
        inputs, targets_raw = dataset[i]
        inputs = inputs.unsqueeze(0).to(device)
        targets_raw = targets_raw.unsqueeze(0).to(device)
        
        target = {
            "albedo": targets_raw[:, 0],
            "roughness": targets_raw[:, 1, 0:1],
            "metallic": targets_raw[:, 2, 0:1],
            "normal": targets_raw[:, 3]
        }
        target = denormalize_target(target, config.transform.mean, config.transform.std)
        
        model.zero_grad()
        outputs = model(inputs)
        
        # Collect output stats
        stats['metallic']['std'].append(outputs['metallic'].std().item())
        stats['roughness']['std'].append(outputs['roughness'].std().item())
        stats['albedo']['std'].append(outputs['albedo'].std().item())
        stats['normal']['z'].append(outputs['normal'][:, 2, :, :].mean().item())
        
        # Compute gradient
        loss, _ = criterion(outputs, target)
        loss.backward()
        
        # Collect gradient stats for each head
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'heads.0.weight' in name:
                    stats['albedo']['grad'].append(param.grad.norm().item())
                elif 'heads.1.weight' in name:
                    stats['roughness']['grad'].append(param.grad.norm().item())
                elif 'heads.2.weight' in name:
                    stats['metallic']['grad'].append(param.grad.norm().item())
                elif 'heads.3.weight' in name:
                    stats['normal']['grad'].append(param.grad.norm().item())
    
    print(f"\n{'='*60}")
    print("OUTPUT STATISTICS")
    print(f"{'='*60}")
    
    for head in ['albedo', 'roughness', 'metallic']:
        avg_std = np.mean(stats[head]['std'])
        avg_grad = np.mean(stats[head]['grad']) if stats[head]['grad'] else 0
        print(f"\n{head.upper()}:")
        print(f"  Output std: {avg_std:.6f}")
        print(f"  Gradient:   {avg_grad:.2e}")
    
    # Normal uses Z-component instead of std
    avg_z = np.mean(stats['normal']['z'])
    std_z = np.std(stats['normal']['z'])
    avg_grad = np.mean(stats['normal']['grad']) if stats['normal']['grad'] else 0
    print(f"\nNORMAL:")
    print(f"  Avg Z:      {avg_z:.4f} (1.0 = flat, want < 0.95)")
    print(f"  Std Z:      {std_z:.4f} (want > 0.01)")
    print(f"  Gradient:   {avg_grad:.2e}")
    
    print(f"\n{'='*60}")
    print("VERDICT:")
    print(f"{'='*60}")
    
    is_healthy = True
    
    # Check metallic
    metallic_std = np.mean(stats['metallic']['std'])
    if metallic_std < 0.001:
        print("  ✗ CRITICAL: Metallic collapsed (std ≈ 0)")
        is_healthy = False
    elif metallic_std < 0.01:
        print(f"  ⚠ WARNING: Low metallic variance (std={metallic_std:.4f})")
    else:
        print(f"  ✓ Metallic OK (std={metallic_std:.4f})")
    
    # Check roughness
    roughness_std = np.mean(stats['roughness']['std'])
    if roughness_std < 0.001:
        print("  ✗ CRITICAL: Roughness collapsed (std ≈ 0)")
        is_healthy = False
    elif roughness_std < 0.01:
        print(f"  ⚠ WARNING: Low roughness variance (std={roughness_std:.4f})")
    else:
        print(f"  ✓ Roughness OK (std={roughness_std:.4f})")
    
    # Check normal
    if avg_z > 0.99 and std_z < 0.01:
        print(f"  ✗ CRITICAL: Normal collapsed to flat (Z={avg_z:.4f})")
        is_healthy = False
    elif avg_z > 0.95:
        print(f"  ⚠ WARNING: Normal mostly flat (Z={avg_z:.4f})")
    else:
        print(f"  ✓ Normal OK (Z={avg_z:.4f})")
    
    # Check albedo
    albedo_std = np.mean(stats['albedo']['std'])
    if albedo_std < 0.01:
        print(f"  ⚠ WARNING: Low albedo variance (std={albedo_std:.4f})")
    else:
        print(f"  ✓ Albedo OK (std={albedo_std:.4f})")
    
    if is_healthy:
        print("\n  → All heads learning, continue training!")
    else:
        print("\n  → ABORT training and investigate!")
    
    return 0 if is_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
