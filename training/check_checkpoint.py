#!/usr/bin/env python3
"""
Analyzes a saved checkpoint to detect signs of model collapse, training instability,
or other issues that would make it unsuitable for inference or shard generation.

Usage:
    python check_checkpoint.py path/to/checkpoint.pth
    python check_checkpoint.py path/to/checkpoint.pth --verbose
    python check_checkpoint.py path/to/checkpoint.pth --test-inference --input-dir ./data/input
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import numpy as np


class HealthStatus(Enum):
    HEALTHY = "✓ HEALTHY"
    WARNING = "⚠ WARNING"
    CRITICAL = "✗ CRITICAL"


@dataclass
class CheckResult:
    name: str
    status: HealthStatus
    message: str
    details: Optional[str] = None


def load_checkpoint(path: str) -> Dict:
    """Load checkpoint with error handling."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch
        ckpt = torch.load(path, map_location="cpu")
    return ckpt


def check_output_head_biases(state_dict: Dict[str, torch.Tensor]) -> List[CheckResult]:
    """
    Check if output head biases are pushed to extremes.
    
    When a model collapses, output biases often get pushed to large positive/negative
    values, causing sigmoid outputs to saturate at 0 or 1.
    """
    results = []
    
    # Find output head layers (typically named 'heads' or 'final_conv')
    head_patterns = ['heads', 'final_conv', 'output', 'head']
    
    bias_keys = [k for k in state_dict.keys() if 'bias' in k.lower()]
    
    for key in bias_keys:
        is_output = any(p in key.lower() for p in head_patterns)
        if not is_output:
            continue
            
        bias = state_dict[key]
        
        # Check for extreme biases
        # For sigmoid outputs: bias > 5 means output > 0.99, bias < -5 means output < 0.01
        max_val = bias.abs().max().item()
        mean_val = bias.mean().item()
        
        if max_val > 10:
            results.append(CheckResult(
                name=f"Output bias: {key}",
                status=HealthStatus.CRITICAL,
                message=f"Extreme bias detected (max={max_val:.2f})",
                details=f"Bias values: min={bias.min().item():.3f}, max={bias.max().item():.3f}, mean={mean_val:.3f}"
            ))
        elif max_val > 5:
            results.append(CheckResult(
                name=f"Output bias: {key}",
                status=HealthStatus.WARNING,
                message=f"High bias detected (max={max_val:.2f})",
                details=f"Bias values: min={bias.min().item():.3f}, max={bias.max().item():.3f}, mean={mean_val:.3f}"
            ))
    
    if not results:
        results.append(CheckResult(
            name="Output head biases",
            status=HealthStatus.HEALTHY,
            message="All output biases within normal range"
        ))
    
    return results


def check_weight_statistics(state_dict: Dict[str, torch.Tensor]) -> List[CheckResult]:
    """Check for NaN, Inf, or degenerate weight distributions."""
    results = []
    
    nan_layers = []
    inf_layers = []
    zero_layers = []
    
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.numel() == 0:
            continue
            
        if torch.isnan(tensor).any():
            nan_layers.append(key)
        if torch.isinf(tensor).any():
            inf_layers.append(key)
        if tensor.numel() > 10 and (tensor == 0).all():
            zero_layers.append(key)
    
    if nan_layers:
        results.append(CheckResult(
            name="NaN weights",
            status=HealthStatus.CRITICAL,
            message=f"Found NaN in {len(nan_layers)} layers",
            details=", ".join(nan_layers[:5]) + ("..." if len(nan_layers) > 5 else "")
        ))
    
    if inf_layers:
        results.append(CheckResult(
            name="Inf weights",
            status=HealthStatus.CRITICAL,
            message=f"Found Inf in {len(inf_layers)} layers",
            details=", ".join(inf_layers[:5]) + ("..." if len(inf_layers) > 5 else "")
        ))
    
    if zero_layers:
        results.append(CheckResult(
            name="Zero weights",
            status=HealthStatus.WARNING,
            message=f"Found {len(zero_layers)} all-zero layers",
            details=", ".join(zero_layers[:5]) + ("..." if len(zero_layers) > 5 else "")
        ))
    
    if not nan_layers and not inf_layers and not zero_layers:
        results.append(CheckResult(
            name="Weight statistics",
            status=HealthStatus.HEALTHY,
            message="No NaN, Inf, or degenerate weights found"
        ))
    
    return results


def check_batchnorm_stats(state_dict: Dict[str, torch.Tensor]) -> List[CheckResult]:
    """Check BatchNorm running statistics for anomalies."""
    results = []
    
    running_mean_keys = [k for k in state_dict.keys() if 'running_mean' in k]
    running_var_keys = [k for k in state_dict.keys() if 'running_var' in k]
    
    abnormal_means = []
    abnormal_vars = []
    
    for key in running_mean_keys:
        mean = state_dict[key]
        if mean.abs().max().item() > 100:
            abnormal_means.append((key, mean.abs().max().item()))
    
    for key in running_var_keys:
        var = state_dict[key]
        if var.max().item() > 1000 or var.min().item() < 1e-10:
            abnormal_vars.append((key, var.min().item(), var.max().item()))
    
    if abnormal_means:
        results.append(CheckResult(
            name="BatchNorm running means",
            status=HealthStatus.WARNING,
            message=f"Found {len(abnormal_means)} layers with extreme means",
            details="; ".join([f"{k}: {v:.1f}" for k, v in abnormal_means[:3]])
        ))
    
    if abnormal_vars:
        results.append(CheckResult(
            name="BatchNorm running vars",
            status=HealthStatus.WARNING,
            message=f"Found {len(abnormal_vars)} layers with extreme variances",
            details="; ".join([f"{k}: [{min_v:.1e}, {max_v:.1e}]" for k, min_v, max_v in abnormal_vars[:3]])
        ))
    
    if not abnormal_means and not abnormal_vars:
        results.append(CheckResult(
            name="BatchNorm statistics",
            status=HealthStatus.HEALTHY,
            message=f"All {len(running_mean_keys)} BatchNorm layers have normal statistics"
        ))
    
    return results


def check_discriminator_health(state_dict: Dict[str, torch.Tensor], is_discriminator_dict: bool = False) -> List[CheckResult]:
    """Check discriminator-specific health indicators.
    
    Args:
        state_dict: Either the full checkpoint or the discriminator's state dict
        is_discriminator_dict: If True, state_dict is already the discriminator's state dict
    """
    results = []
    
    if is_discriminator_dict:
        # state_dict IS the discriminator - use all keys
        disc_keys = list(state_dict.keys())
    else:
        # Find discriminator layers in a combined state dict
        disc_keys = [k for k in state_dict.keys() if any(p in k.lower() for p in ['discriminator', 'disc', 'd_'])]
    
    if not disc_keys:
        # No discriminator in checkpoint
        return [CheckResult(
            name="Discriminator",
            status=HealthStatus.HEALTHY,
            message="No discriminator found (GAN not used or not saved)"
        )]
    
    # Check final layer of discriminator
    final_layer_keys = [k for k in disc_keys if 'weight' in k]
    if final_layer_keys:
        # Get the last conv layer (likely the output)
        last_key = sorted(final_layer_keys)[-1]
        weights = state_dict[last_key]
        
        # Check if weights are near-zero (collapsed) or exploded
        weight_norm = weights.norm().item()
        weight_max = weights.abs().max().item()
        
        if weight_norm < 0.01:
            results.append(CheckResult(
                name="Discriminator output weights",
                status=HealthStatus.CRITICAL,
                message="Discriminator weights near zero (collapsed)",
                details=f"Weight norm: {weight_norm:.6f}"
            ))
        elif weight_max > 100:
            results.append(CheckResult(
                name="Discriminator output weights",
                status=HealthStatus.CRITICAL,
                message="Discriminator weights exploded",
                details=f"Max weight: {weight_max:.2f}"
            ))
    
    if not results:
        results.append(CheckResult(
            name="Discriminator",
            status=HealthStatus.HEALTHY,
            message=f"Discriminator weights look normal ({len(disc_keys)} parameters)"
        ))
    
    return results


def check_training_state(ckpt: Dict) -> List[CheckResult]:
    """Check training state information stored in checkpoint."""
    results = []
    
    # Check epoch
    epoch = ckpt.get('epoch', ckpt.get('current_epoch', None))
    if epoch is not None:
        results.append(CheckResult(
            name="Training epoch",
            status=HealthStatus.HEALTHY,
            message=f"Checkpoint from epoch {epoch}"
        ))
    
    # Check best validation loss
    best_val_loss = ckpt.get('best_val_loss', None)
    if best_val_loss is not None:
        if best_val_loss > 10:
            status = HealthStatus.WARNING
            msg = f"High validation loss: {best_val_loss:.4f}"
        elif best_val_loss < 0:
            status = HealthStatus.WARNING
            msg = f"Negative validation loss: {best_val_loss:.4f} (unusual)"
        else:
            status = HealthStatus.HEALTHY
            msg = f"Best validation loss: {best_val_loss:.4f}"
        
        results.append(CheckResult(
            name="Validation loss",
            status=status,
            message=msg
        ))
    
    # Check config for GAN settings
    config = ckpt.get('config', None)
    if config is not None:
        if isinstance(config, dict):
            loss_config = config.get('loss', {})
            w_gan = loss_config.get('w_gan', 0)
        else:
            # Config object
            w_gan = getattr(getattr(config, 'loss', None), 'w_gan', 0) if hasattr(config, 'loss') else 0
        
        if w_gan > 0:
            results.append(CheckResult(
                name="GAN training",
                status=HealthStatus.HEALTHY,
                message=f"GAN enabled with weight {w_gan}"
            ))
    
    return results


def check_optimizer_state(ckpt: Dict) -> List[CheckResult]:
    """Check optimizer state for signs of instability."""
    results = []
    
    # Check generator optimizer
    g_opt_state = ckpt.get('g_optimizer_state_dict', ckpt.get('optimizer_state_dict', None))
    
    if g_opt_state and 'state' in g_opt_state:
        state = g_opt_state['state']
        
        # Check Adam/AdamW momentum and variance
        max_exp_avg = 0
        max_exp_avg_sq = 0
        
        for param_id, param_state in state.items():
            if 'exp_avg' in param_state:
                max_exp_avg = max(max_exp_avg, param_state['exp_avg'].abs().max().item())
            if 'exp_avg_sq' in param_state:
                max_exp_avg_sq = max(max_exp_avg_sq, param_state['exp_avg_sq'].max().item())
        
        if max_exp_avg > 100:
            results.append(CheckResult(
                name="Optimizer momentum",
                status=HealthStatus.WARNING,
                message=f"High momentum values detected (max: {max_exp_avg:.2f})",
                details="This may indicate training instability"
            ))
        
        if max_exp_avg_sq > 1000:
            results.append(CheckResult(
                name="Optimizer variance",
                status=HealthStatus.WARNING,
                message=f"High variance values detected (max: {max_exp_avg_sq:.2f})",
                details="This may indicate exploding gradients"
            ))
    
    if not results:
        results.append(CheckResult(
            name="Optimizer state",
            status=HealthStatus.HEALTHY,
            message="Optimizer state looks normal"
        ))
    
    return results


def run_test_inference(ckpt: Dict, input_dir: str, num_samples: int = 3) -> List[CheckResult]:
    """Run actual inference to check output distributions."""
    results = []
    
    try:
        # Import here to avoid loading everything if not needed
        sys.path.insert(0, str(Path(__file__).parent))
        from train import MultiViewPBRGenerator
        from train_config import TrainConfig
        from utils.dataset import PBRDataset
        
        # Load config
        config = ckpt.get('config', None)
        if config is None:
            return [CheckResult(
                name="Test inference",
                status=HealthStatus.WARNING,
                message="No config in checkpoint, cannot run inference test"
            )]
        
        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiViewPBRGenerator(config).to(device)
        
        # Load weights
        state_dict = ckpt.get('generator_state_dict', ckpt.get('model_state_dict', None))
        if state_dict is None:
            return [CheckResult(
                name="Test inference",
                status=HealthStatus.WARNING,
                message="No generator state dict in checkpoint"
            )]
        
        # Clean state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('_orig_mod.', '').replace('module.', '')
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # Load a few samples
        input_path = Path(input_dir)
        if not input_path.exists():
            return [CheckResult(
                name="Test inference",
                status=HealthStatus.WARNING,
                message=f"Input directory not found: {input_dir}"
            )]
        
        # Find metadata
        metadata_path = input_path / "render_metadata.json"
        output_dir = input_path.parent / "output"
        
        if not metadata_path.exists() or not output_dir.exists():
            return [CheckResult(
                name="Test inference",
                status=HealthStatus.WARNING,
                message="Could not find metadata or output directory"
            )]
        
        # Create dataset
        dataset = PBRDataset(
            input_dir=str(input_path),
            output_dir=str(output_dir),
            metadata_path=str(metadata_path),
            transform_mean=[0.5, 0.5, 0.5],
            transform_std=[0.5, 0.5, 0.5],
            image_size=(512, 512),  # Use smaller size for quick test
            output_size=(512, 512),
            curriculum_mode=0
        )
        
        # Run inference on a few samples
        output_stats = {
            'albedo': {'min': [], 'max': [], 'std': []},
            'roughness': {'min': [], 'max': [], 'std': []},
            'metallic': {'min': [], 'max': [], 'std': []},
            'normal': {'z_mean': []}
        }
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                inputs, _ = dataset[i]
                inputs = inputs.unsqueeze(0).to(device)
                
                outputs = model(inputs)
                
                for key in ['albedo', 'roughness', 'metallic']:
                    out = outputs[key]
                    output_stats[key]['min'].append(out.min().item())
                    output_stats[key]['max'].append(out.max().item())
                    output_stats[key]['std'].append(out.std().item())
                
                # For normals, check if Z component is reasonable
                normal_z = outputs['normal'][:, 2, :, :].mean().item()
                output_stats['normal']['z_mean'].append(normal_z)
        
        # Analyze output statistics
        for key in ['albedo', 'roughness', 'metallic']:
            stats = output_stats[key]
            avg_std = np.mean(stats['std'])
            avg_min = np.mean(stats['min'])
            avg_max = np.mean(stats['max'])
            
            if avg_std < 0.01:
                results.append(CheckResult(
                    name=f"Output distribution: {key}",
                    status=HealthStatus.CRITICAL,
                    message=f"Near-constant output (std={avg_std:.4f})",
                    details=f"Range: [{avg_min:.3f}, {avg_max:.3f}]"
                ))
            elif avg_std < 0.05:
                results.append(CheckResult(
                    name=f"Output distribution: {key}",
                    status=HealthStatus.WARNING,
                    message=f"Low variance output (std={avg_std:.4f})",
                    details=f"Range: [{avg_min:.3f}, {avg_max:.3f}]"
                ))
            else:
                results.append(CheckResult(
                    name=f"Output distribution: {key}",
                    status=HealthStatus.HEALTHY,
                    message=f"Normal variance (std={avg_std:.4f})",
                    details=f"Range: [{avg_min:.3f}, {avg_max:.3f}]"
                ))
        
        # Check normal map
        avg_z = np.mean(output_stats['normal']['z_mean'])
        if avg_z > 0.95:
            results.append(CheckResult(
                name="Output distribution: normal",
                status=HealthStatus.CRITICAL,
                message=f"Flat normals detected (Z={avg_z:.4f})",
                details="Normal map is outputting [0, 0, 1] everywhere"
            ))
        elif avg_z > 0.8:
            results.append(CheckResult(
                name="Output distribution: normal",
                status=HealthStatus.WARNING,
                message=f"Mostly flat normals (Z={avg_z:.4f})"
            ))
        else:
            results.append(CheckResult(
                name="Output distribution: normal",
                status=HealthStatus.HEALTHY,
                message=f"Normal variance looks good (avg Z={avg_z:.4f})"
            ))
        
    except Exception as e:
        results.append(CheckResult(
            name="Test inference",
            status=HealthStatus.WARNING,
            message=f"Could not run inference test: {str(e)}"
        ))
    
    return results


def print_results(results: List[CheckResult], verbose: bool = False):
    """Print results in a formatted way."""
    # Group by status
    critical = [r for r in results if r.status == HealthStatus.CRITICAL]
    warning = [r for r in results if r.status == HealthStatus.WARNING]
    healthy = [r for r in results if r.status == HealthStatus.HEALTHY]
    
    print("\n" + "=" * 60)
    print("CHECKPOINT HEALTH CHECK RESULTS")
    print("=" * 60)
    
    if critical:
        print(f"\n{HealthStatus.CRITICAL.value} ({len(critical)} issues)")
        print("-" * 40)
        for r in critical:
            print(f"  • {r.name}")
            print(f"    {r.message}")
            if verbose and r.details:
                print(f"    Details: {r.details}")
    
    if warning:
        print(f"\n{HealthStatus.WARNING.value} ({len(warning)} issues)")
        print("-" * 40)
        for r in warning:
            print(f"  • {r.name}")
            print(f"    {r.message}")
            if verbose and r.details:
                print(f"    Details: {r.details}")
    
    if verbose and healthy:
        print(f"\n{HealthStatus.HEALTHY.value} ({len(healthy)} checks passed)")
        print("-" * 40)
        for r in healthy:
            print(f"  • {r.name}: {r.message}")
    
    print("\n" + "=" * 60)
    
    # Overall assessment
    if critical:
        print("OVERALL: ✗ UNSAFE - Do not use for shard generation")
        print("         Model shows signs of collapse or corruption.")
        return False
    elif warning:
        print("OVERALL: ⚠ CAUTION - Review warnings before using")
        print("         Model may have issues but could still work.")
        return True
    else:
        print("OVERALL: ✓ SAFE - Checkpoint looks healthy")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Check NeuroPBR checkpoint health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python check_checkpoint.py checkpoints/best_model.pth
    python check_checkpoint.py checkpoints/best_model.pth --verbose
    python check_checkpoint.py checkpoints/best_model.pth --test-inference --input-dir ./data/input
        """
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--test-inference", action="store_true", 
                        help="Run actual inference to check outputs (slower but more thorough)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory for inference test (required if --test-inference)")
    
    args = parser.parse_args()
    
    if args.test_inference and not args.input_dir:
        print("Error: --input-dir required when using --test-inference")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        ckpt = load_checkpoint(args.checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    
    # Get state dict
    state_dict = ckpt.get('generator_state_dict', ckpt.get('model_state_dict', {}))
    disc_state_dict = ckpt.get('discriminator_state_dict', {})
    
    # Run all checks
    all_results = []
    
    print("\nRunning checks...")
    
    # Basic checks
    all_results.extend(check_training_state(ckpt))
    all_results.extend(check_weight_statistics(state_dict))
    all_results.extend(check_output_head_biases(state_dict))
    all_results.extend(check_batchnorm_stats(state_dict))
    all_results.extend(check_optimizer_state(ckpt))
    
    # Discriminator checks
    if disc_state_dict:
        all_results.extend(check_weight_statistics(disc_state_dict))
        all_results.extend(check_discriminator_health(disc_state_dict, is_discriminator_dict=True))
    
    # Optional inference test
    if args.test_inference:
        print("Running inference test...")
        all_results.extend(run_test_inference(ckpt, args.input_dir))
    
    # Print results
    is_safe = print_results(all_results, args.verbose)
    
    sys.exit(0 if is_safe else 1)


if __name__ == "__main__":
    main()
