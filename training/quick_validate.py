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
    parser.add_argument("--config", type=str, default="configs/no_gan_stable.py")
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
    metallic_stds = []
    metallic_grads = []
    
    import random
    indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
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
        
        metallic_std = outputs['metallic'].std().item()
        metallic_stds.append(metallic_std)
        
        # Compute gradient
        loss, _ = criterion(outputs, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if 'heads.2.weight' in name and param.grad is not None:
                metallic_grads.append(param.grad.norm().item())
                break
    
    avg_std = np.mean(metallic_stds)
    max_std = np.max(metallic_stds)
    avg_grad = np.mean(metallic_grads) if metallic_grads else 0
    max_grad = np.max(metallic_grads) if metallic_grads else 0
    
    print(f"\nMetallic Output Statistics:")
    print(f"  Average std:  {avg_std:.6f}")
    print(f"  Max std:      {max_std:.6f}")
    print(f"\nMetallic Gradient Statistics:")
    print(f"  Average grad: {avg_grad:.2e}")
    print(f"  Max grad:     {max_grad:.2e}")
    
    print(f"\n{'='*60}")
    print("VERDICT:")
    print(f"{'='*60}")
    
    is_healthy = True
    
    if avg_std < 0.001:
        print("  ✗ CRITICAL: Metallic output variance near zero (collapsed)")
        print("    → Model is outputting constant values, not learning")
        is_healthy = False
    elif avg_std < 0.01:
        print("  ⚠ WARNING: Low metallic variance, may be collapsing")
        print(f"    → avg_std={avg_std:.6f}, should be > 0.01")
    else:
        print(f"  ✓ GOOD: Metallic has healthy variance ({avg_std:.4f})")
    
    if avg_grad < 1e-7:
        print("  ✗ CRITICAL: Metallic gradients near zero")
        print("    → Gradients not flowing, training will not improve metallic")
        is_healthy = False
    elif avg_grad < 1e-5:
        print("  ⚠ WARNING: Small metallic gradients")
        print(f"    → avg_grad={avg_grad:.2e}, may learn slowly")
    else:
        print(f"  ✓ GOOD: Metallic gradients healthy ({avg_grad:.2e})")
    
    if is_healthy:
        print("\n  → Continue training, metallic head is learning!")
    else:
        print("\n  → ABORT training and investigate!")
    
    return 0 if is_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
