#!/usr/bin/env python3
"""
Checkpoint Repair Script for NeuroPBR.

Attempts to recover a collapsed model by:
1. Re-initializing collapsed output heads (decoder heads)
2. Resetting BatchNorm running statistics
3. Clearing optimizer momentum/variance state
4. Optionally resetting the discriminator

This preserves the encoder and decoder backbone weights while giving
the output layers a fresh start.

Usage:
    python repair_checkpoint.py checkpoint.pth --output repaired.pth
    python repair_checkpoint.py checkpoint.pth --output repaired.pth --reset-discriminator
    python repair_checkpoint.py checkpoint.pth --output repaired.pth --reset-all-bn
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set
import copy

import torch
import torch.nn as nn


def load_checkpoint(path: str) -> Dict:
    """Load checkpoint with error handling."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt


def find_output_head_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Find keys belonging to output heads."""
    head_patterns = ['heads.0', 'heads.1', 'heads.2', 'heads.3', 'final_conv']
    head_keys = []
    
    for key in state_dict.keys():
        # Match decoder heads
        if any(p in key for p in head_patterns):
            head_keys.append(key)
    
    return head_keys


def find_batchnorm_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    """Find BatchNorm running stat keys."""
    bn_keys = {
        'running_mean': [],
        'running_var': [],
        'num_batches_tracked': []
    }
    
    for key in state_dict.keys():
        if 'running_mean' in key:
            bn_keys['running_mean'].append(key)
        elif 'running_var' in key:
            bn_keys['running_var'].append(key)
        elif 'num_batches_tracked' in key:
            bn_keys['num_batches_tracked'].append(key)
    
    return bn_keys


def find_collapsed_bn_keys(state_dict: Dict[str, torch.Tensor], 
                           var_threshold: float = 1e-10) -> List[str]:
    """Find BatchNorm layers with near-zero variance (collapsed)."""
    collapsed = []
    
    for key in state_dict.keys():
        if 'running_var' in key:
            var = state_dict[key]
            if var.min().item() < var_threshold:
                # Find corresponding mean and num_batches
                base = key.replace('running_var', '')
                collapsed.append(base)
    
    return collapsed


def reinit_conv_weight(weight: torch.Tensor) -> torch.Tensor:
    """Re-initialize a conv weight tensor using Kaiming initialization."""
    new_weight = torch.empty_like(weight)
    nn.init.kaiming_normal_(new_weight, mode='fan_out', nonlinearity='relu')
    return new_weight


def reinit_conv_bias(bias: torch.Tensor, output_type: str = 'default') -> torch.Tensor:
    """Re-initialize a conv bias tensor.
    
    Args:
        bias: Original bias tensor
        output_type: 'albedo', 'roughness', 'metallic', 'normal', or 'default'
    """
    new_bias = torch.zeros_like(bias)
    
    if output_type == 'normal':
        # Bias toward [0, 0, 1] for flat normal (good starting point)
        if bias.shape[0] == 3:
            new_bias[2] = 1.0  # Z component
    elif output_type == 'roughness':
        # Bias toward 0.5 (middle value)
        new_bias.fill_(0.0)  # sigmoid(0) = 0.5
    elif output_type == 'metallic':
        # Bias toward 0 (most materials are non-metallic)
        new_bias.fill_(-2.0)  # sigmoid(-2) ≈ 0.12
    elif output_type == 'albedo':
        # Bias toward middle gray
        new_bias.fill_(0.0)  # sigmoid(0) = 0.5
    
    return new_bias


def repair_checkpoint(
    ckpt: Dict,
    reset_heads: bool = True,
    reset_collapsed_bn: bool = True,
    reset_all_bn: bool = False,
    reset_optimizer: bool = True,
    reset_discriminator: bool = False
) -> Dict:
    """
    Repair a collapsed checkpoint.
    
    Args:
        ckpt: Loaded checkpoint dictionary
        reset_heads: Re-initialize output heads (decoder heads)
        reset_collapsed_bn: Reset only BN layers with near-zero variance
        reset_all_bn: Reset ALL BatchNorm running statistics
        reset_optimizer: Clear optimizer momentum/variance
        reset_discriminator: Re-initialize discriminator weights
    
    Returns:
        Repaired checkpoint dictionary
    """
    repaired = copy.deepcopy(ckpt)
    state_dict = repaired.get('generator_state_dict', repaired.get('model_state_dict', {}))
    
    repairs_made = []
    
    # 1. Reset output heads
    if reset_heads:
        head_keys = find_output_head_keys(state_dict)
        
        # Determine output types based on head index
        # heads.0 = albedo (3ch), heads.1 = roughness (1ch), 
        # heads.2 = metallic (1ch), heads.3 = normal (3ch)
        output_types = {
            'heads.0': 'albedo',
            'heads.1': 'roughness', 
            'heads.2': 'metallic',
            'heads.3': 'normal'
        }
        
        for key in head_keys:
            # Determine output type
            out_type = 'default'
            for pattern, otype in output_types.items():
                if pattern in key:
                    out_type = otype
                    break
            
            if 'weight' in key:
                old_weight = state_dict[key]
                state_dict[key] = reinit_conv_weight(old_weight)
                repairs_made.append(f"Re-initialized weight: {key}")
            elif 'bias' in key:
                old_bias = state_dict[key]
                state_dict[key] = reinit_conv_bias(old_bias, out_type)
                repairs_made.append(f"Re-initialized bias: {key} (type={out_type})")
    
    # 2. Reset BatchNorm statistics
    bn_keys = find_batchnorm_keys(state_dict)
    
    if reset_all_bn:
        # Reset ALL BatchNorm layers
        for key in bn_keys['running_mean']:
            state_dict[key] = torch.zeros_like(state_dict[key])
        for key in bn_keys['running_var']:
            state_dict[key] = torch.ones_like(state_dict[key])
        for key in bn_keys['num_batches_tracked']:
            state_dict[key] = torch.tensor(0)
        repairs_made.append(f"Reset ALL {len(bn_keys['running_mean'])} BatchNorm layers")
    
    elif reset_collapsed_bn:
        # Reset only collapsed BatchNorm layers
        collapsed_bases = find_collapsed_bn_keys(state_dict)
        
        for base in collapsed_bases:
            mean_key = base + 'running_mean'
            var_key = base + 'running_var'
            track_key = base + 'num_batches_tracked'
            
            if mean_key in state_dict:
                state_dict[mean_key] = torch.zeros_like(state_dict[mean_key])
            if var_key in state_dict:
                state_dict[var_key] = torch.ones_like(state_dict[var_key])
            if track_key in state_dict:
                state_dict[track_key] = torch.tensor(0)
        
        if collapsed_bases:
            repairs_made.append(f"Reset {len(collapsed_bases)} collapsed BatchNorm layers")
    
    # 3. Reset optimizer state
    if reset_optimizer:
        # Clear generator optimizer momentum
        if 'g_optimizer_state_dict' in repaired:
            g_opt = repaired['g_optimizer_state_dict']
            if 'state' in g_opt:
                # Clear all parameter states (momentum, variance, etc)
                g_opt['state'] = {}
                repairs_made.append("Cleared generator optimizer state")
        
        # Also try the generic key
        if 'optimizer_state_dict' in repaired:
            opt = repaired['optimizer_state_dict']
            if 'state' in opt:
                opt['state'] = {}
                repairs_made.append("Cleared optimizer state")
    
    # 4. Reset discriminator
    if reset_discriminator and 'discriminator_state_dict' in repaired:
        disc_state = repaired['discriminator_state_dict']
        
        for key in disc_state.keys():
            if 'weight' in key and isinstance(disc_state[key], torch.Tensor):
                if disc_state[key].dim() >= 2:  # Conv or Linear
                    disc_state[key] = reinit_conv_weight(disc_state[key])
            elif 'bias' in key and isinstance(disc_state[key], torch.Tensor):
                disc_state[key] = torch.zeros_like(disc_state[key])
        
        # Reset discriminator optimizer
        if 'd_optimizer_state_dict' in repaired:
            d_opt = repaired['d_optimizer_state_dict']
            if 'state' in d_opt:
                d_opt['state'] = {}
        
        repairs_made.append("Re-initialized discriminator weights and optimizer")
    
    # 5. Reset training state
    repaired['epoch'] = 0
    repaired['global_step'] = 0
    repaired['best_val_loss'] = float('inf')
    repairs_made.append("Reset epoch to 0, cleared best_val_loss")
    
    # 6. Update config to disable GAN for first epochs (give reconstruction time to stabilize)
    config = repaired.get('config', None)
    if config is not None:
        if hasattr(config, 'training'):
            config.training.gan_start_epoch = 10  # Delay GAN
            repairs_made.append("Set gan_start_epoch to 10 (delayed GAN)")
        elif isinstance(config, dict) and 'training' in config:
            config['training']['gan_start_epoch'] = 10
            repairs_made.append("Set gan_start_epoch to 10 (delayed GAN)")
    
    # Update state dict in checkpoint
    if 'generator_state_dict' in repaired:
        repaired['generator_state_dict'] = state_dict
    elif 'model_state_dict' in repaired:
        repaired['model_state_dict'] = state_dict
    
    return repaired, repairs_made


def main():
    parser = argparse.ArgumentParser(
        description="Repair a collapsed NeuroPBR checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic repair (reset heads, collapsed BN, optimizer)
    python repair_checkpoint.py checkpoint.pth --output repaired.pth
    
    # Also reset discriminator (recommended if GAN was unstable)
    python repair_checkpoint.py checkpoint.pth --output repaired.pth --reset-discriminator
    
    # Nuclear option: reset ALL BatchNorm layers
    python repair_checkpoint.py checkpoint.pth --output repaired.pth --reset-all-bn

After repair, resume training:
    python train.py --resume repaired.pth --input-dir ./data/input --output-dir ./data/output
        """
    )
    
    parser.add_argument("checkpoint", type=str, help="Path to collapsed checkpoint")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path for repaired checkpoint")
    parser.add_argument("--reset-discriminator", action="store_true",
                        help="Also re-initialize discriminator weights")
    parser.add_argument("--reset-all-bn", action="store_true",
                        help="Reset ALL BatchNorm layers (not just collapsed ones)")
    parser.add_argument("--no-reset-heads", action="store_true",
                        help="Don't re-initialize output heads")
    parser.add_argument("--no-reset-optimizer", action="store_true",
                        help="Don't clear optimizer state")
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        ckpt = load_checkpoint(args.checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Original epoch: {ckpt.get('epoch', 'unknown')}")
    print()
    
    # Repair
    print("Repairing checkpoint...")
    repaired, repairs = repair_checkpoint(
        ckpt,
        reset_heads=not args.no_reset_heads,
        reset_collapsed_bn=not args.reset_all_bn,
        reset_all_bn=args.reset_all_bn,
        reset_optimizer=not args.no_reset_optimizer,
        reset_discriminator=args.reset_discriminator
    )
    
    print()
    print("Repairs made:")
    for r in repairs:
        print(f"  • {r}")
    
    # Save
    print()
    print(f"Saving repaired checkpoint to: {args.output}")
    torch.save(repaired, args.output)
    
    print()
    print("=" * 60)
    print("REPAIR COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Verify the repair:")
    print(f"     python check_checkpoint.py {args.output} --test-inference --input-dir /path/to/input --output-dir /path/to/output")
    print()
    print("  2. Resume training (GAN delayed to epoch 10):")
    print(f"     python train.py --resume {args.output} --input-dir ./data/input --output-dir ./data/output")
    print()
    print("  3. Watch for signs of recovery in first few epochs:")
    print("     - Loss should drop quickly in epochs 1-5")
    print("     - Albedo/normal std should increase (model producing varied outputs)")
    print("     - If still collapsed after 5 epochs, start fresh")
    print()


if __name__ == "__main__":
    main()
