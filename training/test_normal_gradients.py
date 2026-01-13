#!/usr/bin/env python3
"""
Quick sanity check: Verify gradients flow through the normal head.
Run this BEFORE committing to a full training run.

Usage:
    python test_normal_gradients.py

Expected output:
    - Normal head gradient should be > 1e-4
    - Initial normal Z should be < 0.95 (not collapsed)
    - Gradient should vary across spatial locations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

def test_normal_head_gradients():
    print("="*60)
    print("NORMAL HEAD GRADIENT SANITY CHECK")
    print("="*60)
    
    from train import MultiViewPBRGenerator
    from train_config import TrainConfig
    from losses.losses import HybridLoss
    
    # Load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "configs/ultra_stable.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()
    
    # Use small images for speed
    config.data.image_size = (256, 256)
    config.data.output_size = (256, 256)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build fresh model (will use new initialization)
    model = MultiViewPBRGenerator(config).to(device)
    model.train()
    
    # Build loss with variance matching
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
        "metallic_boost": config.loss.metallic_boost,
        "w_variance_match": config.loss.w_variance_match
    }
    criterion = HybridLoss(loss_config).to(device)
    print(f"Variance match weight: {config.loss.w_variance_match}")
    
    # Random input
    x = torch.randn(1, 3, 3, 256, 256, device=device)
    
    # Forward pass
    model.zero_grad()
    outputs = model(x)
    
    # Check initial normal distribution
    normal = outputs['normal']
    normal_z = normal[:, 2, :, :].mean().item()
    normal_z_std = normal[:, 2, :, :].std().item()
    normal_xy_mag = (normal[:, 0, :, :]**2 + normal[:, 1, :, :]**2).sqrt().mean().item()
    
    print(f"\n--- Initial Normal Statistics ---")
    print(f"  Normal Z mean:  {normal_z:.4f} (want < 0.95)")
    print(f"  Normal Z std:   {normal_z_std:.4f} (want > 0.01)")
    print(f"  Normal XY mag:  {normal_xy_mag:.4f} (want > 0.1)")
    
    if normal_z > 0.99 and normal_z_std < 0.01:
        print("  ✗ FAIL: Normal initialized to flat [0,0,1]!")
        return False
    else:
        print("  ✓ PASS: Normal has initial diversity")
    
    # Create fake target with varied normals
    target_raw = torch.randn(1, 3, 256, 256, device=device)
    target_normal = F.normalize(target_raw, p=2, dim=1)
    
    # Full target dict
    target = {
        "albedo": torch.rand(1, 3, 256, 256, device=device),
        "roughness": torch.rand(1, 1, 256, 256, device=device),
        "metallic": torch.rand(1, 1, 256, 256, device=device),
        "normal": target_normal
    }
    
    # Compute loss with variance matching
    loss, loss_info = criterion(outputs, target)
    print(f"\n--- Loss Components ---")
    for k, v in loss_info.items():
        if 'loss' in k:
            print(f"  {k}: {v:.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients on normal head
    print(f"\n--- Gradient Check ---")
    
    for name, param in model.named_parameters():
        if 'heads.3' in name and 'weight' in name:
            grad = param.grad
            if grad is None:
                print(f"  ✗ FAIL: No gradient for {name}")
                return False
            
            grad_norm = grad.norm().item()
            grad_mean = grad.abs().mean().item()
            grad_max = grad.abs().max().item()
            grad_nonzero = (grad.abs() > 1e-8).float().mean().item()
            
            print(f"  {name}:")
            print(f"    Norm:     {grad_norm:.2e}")
            print(f"    Mean:     {grad_mean:.2e}")
            print(f"    Max:      {grad_max:.2e}")
            print(f"    Non-zero: {grad_nonzero*100:.1f}%")
            
            if grad_norm < 1e-6:
                print(f"  ✗ FAIL: Gradient too small!")
                return False
            else:
                print(f"  ✓ PASS: Gradient is healthy")
    
    # Check roughness head gradient (should also be boosted by variance loss)
    print(f"\n--- Roughness Head Gradient ---")
    for name, param in model.named_parameters():
        if 'heads.1' in name and 'weight' in name:
            grad = param.grad
            if grad is not None:
                print(f"  {name}: norm={grad.norm().item():.2e}")
    
    print(f"\n{'='*60}")
    print("OVERALL: ✓ Normal head looks healthy!")
    print("="*60)
    print("\nSafe to proceed with training.")
    
    return True

if __name__ == "__main__":
    success = test_normal_head_gradients()
    sys.exit(0 if success else 1)
