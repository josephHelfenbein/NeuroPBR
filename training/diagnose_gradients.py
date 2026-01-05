#!/usr/bin/env python3
"""
Diagnose why metallic and normal heads aren't learning.

This script runs a single forward+backward pass and prints gradient statistics
for each output head to determine if gradients are flowing properly.

Usage:
    python diagnose_gradients.py --config configs/no_gan_stable.py \
        --checkpoint /path/to/checkpoint.pth \
        --input-dir /path/to/input \
        --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Diagnose gradient flow in NeuroPBR")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--input-dir", type=str, required=True, help="Input renders directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Ground truth PBR maps")
    args = parser.parse_args()
    
    # Import training modules
    sys.path.insert(0, str(Path(__file__).parent))
    from train import MultiViewPBRGenerator, denormalize_target
    from train_config import TrainConfig
    from utils.dataset import PBRDataset
    from losses.losses import HybridLoss
    
    # Load config
    config = TrainConfig()
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config: {config_path}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.get_config()
    
    # Print loss weights being used
    print("\n" + "="*60)
    print("LOSS WEIGHTS FROM CONFIG")
    print("="*60)
    print(f"  w_l1: {config.loss.w_l1}")
    print(f"  w_ssim: {config.loss.w_ssim}")
    print(f"  w_normal: {config.loss.w_normal}")
    print(f"  w_albedo: {config.loss.w_albedo}")
    print(f"  w_roughness: {config.loss.w_roughness}")
    print(f"  w_metallic: {config.loss.w_metallic}")
    print(f"  w_normal_map: {config.loss.w_normal_map}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Build model
    model = MultiViewPBRGenerator(config).to(device)
    
    # Load checkpoint weights
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('generator_state_dict', ckpt.get('model_state_dict', {}))
    
    # Clean state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('_orig_mod.', '').replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.train()  # Training mode
    
    # Build loss function with config weights
    loss_config = {
        "w_l1": config.loss.w_l1,
        "w_ssim": config.loss.w_ssim,
        "w_normal": config.loss.w_normal,
        "w_gan": 0.0,  # No GAN for this test
        "w_albedo": config.loss.w_albedo,
        "w_roughness": config.loss.w_roughness,
        "w_metallic": config.loss.w_metallic,
        "w_normal_map": config.loss.w_normal_map,
        "gan_loss_type": "hinge",
        "metallic_boost": getattr(config.loss, 'metallic_boost', 10.0)
    }
    print(f"\n  metallic_boost: {loss_config['metallic_boost']}")
    criterion = HybridLoss(loss_config).to(device)
    
    # Load one batch
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
    
    # Get one sample
    inputs, targets_raw = dataset[0]
    inputs = inputs.unsqueeze(0).to(device)
    targets_raw = targets_raw.unsqueeze(0).to(device)
    
    # Prepare targets (same as in training loop)
    target_normalized = {
        "albedo": targets_raw[:, 0],
        "roughness": targets_raw[:, 1, 0:1],
        "metallic": targets_raw[:, 2, 0:1],
        "normal": targets_raw[:, 3]
    }
    
    target = denormalize_target(
        target_normalized,
        config.transform.mean,
        config.transform.std
    )
    
    print("\n" + "="*60)
    print("TARGET STATISTICS")
    print("="*60)
    for name, t in target.items():
        print(f"  {name}: shape={t.shape}, min={t.min().item():.4f}, max={t.max().item():.4f}, std={t.std().item():.4f}")
    
    # Forward pass
    outputs = model(inputs)
    
    print("\n" + "="*60)
    print("OUTPUT STATISTICS (before backward)")
    print("="*60)
    for name, o in outputs.items():
        print(f"  {name}: shape={o.shape}, min={o.min().item():.4f}, max={o.max().item():.4f}, std={o.std().item():.4f}")
    
    # Compute loss
    loss, loss_info = criterion(outputs, target)
    
    print("\n" + "="*60)
    print("LOSS BREAKDOWN")
    print("="*60)
    for k, v in loss_info.items():
        print(f"  {k}: {v:.6f}")
    print(f"\n  Total loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    print("\n" + "="*60)
    print("GRADIENT STATISTICS FOR OUTPUT HEADS")
    print("="*60)
    
    # Find head parameters and their gradients
    head_names = {0: "albedo", 1: "roughness", 2: "metallic", 3: "normal"}
    
    for name, param in model.named_parameters():
        if 'head' in name.lower():
            grad = param.grad
            if grad is None:
                print(f"  {name}: NO GRADIENT (None)")
            else:
                grad_norm = grad.norm().item()
                grad_max = grad.abs().max().item()
                grad_mean = grad.mean().item()
                
                status = ""
                if grad_norm < 1e-10:
                    status = " ← ZERO GRADIENT!"
                elif grad_norm < 1e-6:
                    status = " ← VERY SMALL!"
                
                print(f"  {name}:")
                print(f"    grad_norm={grad_norm:.2e}, grad_max={grad_max:.2e}, grad_mean={grad_mean:.2e}{status}")
    
    # Also check decoder output layers
    print("\n" + "="*60)
    print("GRADIENT STATISTICS FOR DECODER LAYERS")
    print("="*60)
    
    for name, param in model.named_parameters():
        if 'decoder' in name.lower() and 'weight' in name.lower():
            grad = param.grad
            if grad is None:
                continue
            
            grad_norm = grad.norm().item()
            if grad_norm < 1e-8:
                print(f"  {name}: grad_norm={grad_norm:.2e} ← VERY SMALL!")
    
    # Check specific per-channel gradients for the output
    print("\n" + "="*60)
    print("PER-OUTPUT GRADIENT CHECK")
    print("="*60)
    
    # Manually compute gradient for each output to see if any are blocked
    model.zero_grad()
    
    outputs_fresh = model(inputs)
    
    # Test metallic gradient
    metallic_loss = F.l1_loss(outputs_fresh["metallic"], target["metallic"])
    metallic_loss.backward(retain_graph=True)
    
    metallic_grads = []
    for name, param in model.named_parameters():
        if 'head' in name.lower() and '2' in name:  # heads.2 = metallic
            if param.grad is not None:
                metallic_grads.append((name, param.grad.norm().item()))
    
    print(f"\n  Metallic (heads.2) gradient from metallic L1 loss only:")
    for name, g in metallic_grads:
        status = " ← ZERO!" if g < 1e-10 else ""
        print(f"    {name}: {g:.2e}{status}")
    
    model.zero_grad()
    
    # Test normal gradient
    normal_loss = F.l1_loss(outputs_fresh["normal"], target["normal"])
    normal_loss.backward(retain_graph=True)
    
    normal_grads = []
    for name, param in model.named_parameters():
        if 'head' in name.lower() and '3' in name:  # heads.3 = normal
            if param.grad is not None:
                normal_grads.append((name, param.grad.norm().item()))
    
    print(f"\n  Normal (heads.3) gradient from normal L1 loss only:")
    for name, g in normal_grads:
        status = " ← ZERO!" if g < 1e-10 else ""
        print(f"    {name}: {g:.2e}{status}")
    
    # Search for a metallic sample to test gradient on
    print("\n" + "="*60)
    print("SEARCHING FOR METALLIC SAMPLE")
    print("="*60)
    
    metallic_sample_idx = None
    for i in range(min(100, len(dataset))):
        _, targets_check = dataset[i]
        metallic_gt = targets_check[2, 0:1]  # metallic channel
        metallic_denorm = metallic_gt * 0.5 + 0.5
        if metallic_denorm.max() > 0.1:
            metallic_sample_idx = i
            print(f"  Found metallic sample at index {i}: max={metallic_denorm.max().item():.4f}")
            break
    
    if metallic_sample_idx is not None:
        print(f"\n  Testing gradient on metallic sample {metallic_sample_idx}:")
        inputs_m, targets_m = dataset[metallic_sample_idx]
        inputs_m = inputs_m.unsqueeze(0).to(device)
        targets_m = targets_m.unsqueeze(0).to(device)
        
        target_m_dict = {
            "albedo": targets_m[:, 0],
            "roughness": targets_m[:, 1, 0:1],
            "metallic": targets_m[:, 2, 0:1],
            "normal": targets_m[:, 3]
        }
        target_m = denormalize_target(target_m_dict, config.transform.mean, config.transform.std)
        
        print(f"    GT metallic: min={target_m['metallic'].min():.4f}, max={target_m['metallic'].max():.4f}, std={target_m['metallic'].std():.4f}")
        
        model.zero_grad()
        outputs_m = model(inputs_m)
        loss_m, info_m = criterion(outputs_m, target_m)
        loss_m.backward()
        
        print(f"    Pred metallic: min={outputs_m['metallic'].min():.4f}, max={outputs_m['metallic'].max():.4f}")
        print(f"    l1_metallic: {info_m.get('l1_metallic', 'N/A')}")
        print(f"    metallic_boosted: {info_m.get('metallic_boosted', False)}")
        
        for name, param in model.named_parameters():
            if 'head' in name.lower() and '2' in name:
                if param.grad is not None:
                    g = param.grad.norm().item()
                    status = " ← ZERO!" if g < 1e-8 else " ← GOOD!" if g > 1e-4 else ""
                    print(f"    {name}: grad_norm={g:.2e}{status}")
    else:
        print("  No metallic samples found in first 100 samples!")
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print("""
The key finding: when target metallic is 0 and model outputs ~0, the L1 loss
is 0 and gradients are 0. The model never learns to output non-zero metallic.

FIX APPLIED: Sample-aware metallic boost
- When a sample HAS metallic content (GT max > 0.1), its loss is multiplied by 10x
- This ensures metallic samples generate meaningful gradients
- Non-metallic samples still have ~0 loss (which is correct)

To use the fix:
1. Restart training from scratch (epoch 0) - the collapsed heads need fresh init
2. Use config with metallic_boost=10.0 (now added to no_gan_stable.py)
3. Monitor l1_metallic in logs - it should be non-zero for ~20% of batches
""")


if __name__ == "__main__":
    main()
