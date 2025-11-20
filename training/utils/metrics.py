"""
Metrics for evaluating PBR map quality.

Includes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Angular error for normal maps
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between prediction and target.
    
    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        max_val: Maximum possible pixel value (1.0 for normalized images)
        eps: Small constant for numerical stability
        
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse < eps:
        return 100.0  # Effectively perfect
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Calculate SSIM (Structural Similarity Index) between prediction and target.
    
    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        window_size: Size of Gaussian window
        sigma: Standard deviation of Gaussian window
        
    Returns:
        SSIM value [0, 1] where 1 is perfect similarity
    """
    device = pred.device
    channel = pred.shape[1]
    
    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window_2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    
    # SSIM constants
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Calculate means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    
    return ssim_map.mean().item()


def angular_error(pred: torch.Tensor, target: torch.Tensor, normalize: bool = True, eps: float = 1e-8) -> Dict[str, float]:
    """
    Calculate angular error between predicted and target normal maps.
    
    Args:
        pred: Predicted normal map (B, 3, H, W)
        target: Target normal map (B, 3, H, W)
        normalize: Whether to normalize normals before comparison
        eps: Small constant for numerical stability
        
    Returns:
        Dictionary with mean, median, and other statistics in degrees
    """
    if normalize:
        pred = F.normalize(pred, p=2, dim=1, eps=eps)
        target = F.normalize(target, p=2, dim=1, eps=eps)
    
    # Compute cosine similarity
    cos_sim = (pred * target).sum(dim=1, keepdim=True)
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    
    # Convert to angles in degrees
    angles_rad = torch.acos(cos_sim)
    angles_deg = angles_rad * 180.0 / np.pi
    
    # Flatten for statistics
    angles_flat = angles_deg.flatten().cpu().numpy()
    
    return {
        "mean": float(np.mean(angles_flat)),
        "median": float(np.median(angles_flat)),
        "std": float(np.std(angles_flat)),
        "min": float(np.min(angles_flat)),
        "max": float(np.max(angles_flat)),
        "percentile_90": float(np.percentile(angles_flat, 90)),
        "percentile_95": float(np.percentile(angles_flat, 95))
    }


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        
    Returns:
        MAE value
    """
    return F.l1_loss(pred, target).item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        
    Returns:
        RMSE value
    """
    return torch.sqrt(F.mse_loss(pred, target)).item()


def compute_pbr_metrics(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    include_angular: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for all PBR maps.
    
    Args:
        pred: Dictionary with predicted PBR maps (albedo, roughness, metallic, normal)
        target: Dictionary with target PBR maps
        include_angular: Whether to compute angular error (slower)
        
    Returns:
        Dictionary of all computed metrics
    """
    metrics = {}
    
    # Metrics for each map type
    for map_name in ["albedo", "roughness", "metallic"]:
        if map_name in pred and map_name in target:
            prefix = map_name
            metrics[f"{prefix}_psnr"] = psnr(pred[map_name], target[map_name])
            metrics[f"{prefix}_ssim"] = ssim(pred[map_name], target[map_name])
            metrics[f"{prefix}_mae"] = mae(pred[map_name], target[map_name])
            metrics[f"{prefix}_rmse"] = rmse(pred[map_name], target[map_name])
    
    # Normal map - angular error
    if "normal" in pred and "normal" in target:
        if include_angular:
            angular_stats = angular_error(pred["normal"], target["normal"])
            for key, val in angular_stats.items():
                metrics[f"normal_angle_{key}"] = val
        
        # Standard metrics for normal as well
        metrics["normal_mae"] = mae(pred["normal"], target["normal"])
        metrics["normal_rmse"] = rmse(pred["normal"], target["normal"])
    
    # Overall metrics (average across all maps)
    psnr_vals = [v for k, v in metrics.items() if k.endswith("_psnr")]
    ssim_vals = [v for k, v in metrics.items() if k.endswith("_ssim")]
    mae_vals = [v for k, v in metrics.items() if k.endswith("_mae")]
    
    if psnr_vals:
        metrics["overall_psnr"] = sum(psnr_vals) / len(psnr_vals)
    if ssim_vals:
        metrics["overall_ssim"] = sum(ssim_vals) / len(ssim_vals)
    if mae_vals:
        metrics["overall_mae"] = sum(mae_vals) / len(mae_vals)
    
    return metrics


# Backward compatibility
def calculate_metrics(pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Legacy name for compute_pbr_metrics."""
    return compute_pbr_metrics(pred, target, include_angular=True)


if __name__ == "__main__":
    # Test metrics
    print("Testing PBR metrics...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    pred = {
        "albedo": torch.rand(2, 3, 256, 256, device=device),
        "roughness": torch.rand(2, 1, 256, 256, device=device),
        "metallic": torch.rand(2, 1, 256, 256, device=device),
        "normal": F.normalize(torch.randn(2, 3, 256, 256, device=device), p=2, dim=1)
    }
    
    target = {
        "albedo": pred["albedo"] + 0.1 * torch.randn_like(pred["albedo"]),
        "roughness": pred["roughness"] + 0.1 * torch.randn_like(pred["roughness"]),
        "metallic": pred["metallic"] + 0.1 * torch.randn_like(pred["metallic"]),
        "normal": F.normalize(pred["normal"] + 0.1 * torch.randn_like(pred["normal"]), p=2, dim=1)
    }
    
    # Compute metrics
    metrics = compute_pbr_metrics(pred, target)
    
    print("\nComputed metrics:")
    for key, val in sorted(metrics.items()):
        print(f"  {key}: {val:.4f}")
    
    print("\n✓ Metrics module working correctly!")
