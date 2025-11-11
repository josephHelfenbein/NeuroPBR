"""
Visualization utilities for PBR maps and training progress.

Includes:
- TensorBoard image logging
- Side-by-side comparisons
- Error maps
- Normal map visualizations
"""

import torch
import torch.nn.functional as F
import torchvision
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter


def normalize_for_display(tensor: torch.Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> torch.Tensor:
    """
    Normalize tensor to [0, 1] range for display.
    
    Args:
        tensor: Input tensor
        min_val: Minimum value (if None, use tensor min)
        max_val: Maximum value (if None, use tensor max)
        
    Returns:
        Normalized tensor
    """
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
    
    if max_val - min_val < 1e-8:
        return torch.zeros_like(tensor)
    
    return (tensor - min_val) / (max_val - min_val)


def normal_to_rgb(normal: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Convert normal map from [-1, 1] to RGB [0, 1] for visualization.
    
    Args:
        normal: Normal map (B, 3, H, W) with values in [-1, 1]
        normalize: Whether to normalize normals first
        
    Returns:
        RGB tensor (B, 3, H, W) with values in [0, 1]
    """
    if normalize:
        normal = F.normalize(normal, p=2, dim=1, eps=1e-8)
    
    # Convert from [-1, 1] to [0, 1]
    rgb = (normal + 1.0) / 2.0
    return torch.clamp(rgb, 0.0, 1.0)


def create_error_map(pred: torch.Tensor, target: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    """
    Create a heatmap showing per-pixel error.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        scale: Error scaling factor for visualization
        
    Returns:
        RGB error map (B, 3, H, W)
    """
    # Compute absolute error
    error = torch.abs(pred - target).mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Scale and clamp
    error = error * scale
    error = torch.clamp(error, 0.0, 1.0)
    
    # Convert to heatmap (grayscale to RGB, could use colormap)
    # Simple red heatmap: (error, 0, 1-error)
    heatmap = torch.cat([
        error,
        torch.zeros_like(error),
        1.0 - error
    ], dim=1)
    
    return heatmap


def create_comparison_grid(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    max_images: int = 4,
    include_error: bool = True
) -> torch.Tensor:
    """
    Create a comparison grid showing predicted vs target PBR maps.
    
    Layout per sample:
        Pred Albedo | Target Albedo | Error
        Pred Rough  | Target Rough  | Error
        Pred Metal  | Target Metal  | Error
        Pred Normal | Target Normal | Error
    
    Args:
        pred: Dictionary of predicted PBR maps
        target: Dictionary of target PBR maps
        max_images: Maximum number of batch samples to show
        include_error: Whether to include error maps
        
    Returns:
        Grid tensor suitable for tensorboard
    """
    batch_size = pred["albedo"].shape[0]
    num_images = min(batch_size, max_images)
    
    rows = []
    
    # Process each map type
    map_types = ["albedo", "roughness", "metallic", "normal"]
    
    for map_name in map_types:
        if map_name not in pred or map_name not in target:
            continue
        
        pred_map = pred[map_name][:num_images]
        target_map = target[map_name][:num_images]
        
        # Handle single-channel maps (roughness, metallic)
        if pred_map.shape[1] == 1:
            pred_map = pred_map.repeat(1, 3, 1, 1)
            target_map = target_map.repeat(1, 3, 1, 1)
        
        # Normalize normal maps for display
        if map_name == "normal":
            pred_map = normal_to_rgb(pred_map)
            target_map = normal_to_rgb(target_map)
        
        # Create row: [pred1, pred2, ..., target1, target2, ...]
        if include_error:
            error_map = create_error_map(pred[map_name][:num_images], target[map_name][:num_images])
            if error_map.shape[1] == 1:
                error_map = error_map.repeat(1, 3, 1, 1)
            row = torch.cat([pred_map, target_map, error_map], dim=0)
        else:
            row = torch.cat([pred_map, target_map], dim=0)
        
        rows.append(row)
    
    # Stack all rows
    all_images = torch.cat(rows, dim=0)
    
    # Create grid
    nrow = num_images if not include_error else num_images
    grid = torchvision.utils.make_grid(
        all_images,
        nrow=num_images * (3 if include_error else 2),
        normalize=False,
        padding=2,
        pad_value=1.0
    )
    
    return grid


def log_images_to_tensorboard(
    writer: SummaryWriter,
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    global_step: int,
    prefix: str = "val",
    max_images: int = 4
):
    """
    Log PBR map comparisons to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        pred: Dictionary of predicted PBR maps
        target: Dictionary of target PBR maps
        global_step: Current training step
        prefix: Prefix for tensorboard tags (e.g., 'train', 'val')
        max_images: Maximum number of images to log
    """
    # Create comparison grid
    grid = create_comparison_grid(pred, target, max_images=max_images, include_error=True)
    writer.add_image(f"{prefix}/comparison", grid, global_step)
    
    # Log individual maps separately for detail
    num_images = min(pred["albedo"].shape[0], max_images)
    
    for map_name in ["albedo", "roughness", "metallic", "normal"]:
        if map_name not in pred or map_name not in target:
            continue
        
        pred_map = pred[map_name][:num_images]
        target_map = target[map_name][:num_images]
        
        # Handle visualization
        if map_name == "normal":
            pred_map = normal_to_rgb(pred_map)
            target_map = normal_to_rgb(target_map)
        elif pred_map.shape[1] == 1:
            pred_map = pred_map.repeat(1, 3, 1, 1)
            target_map = target_map.repeat(1, 3, 1, 1)
        
        # Make grids
        pred_grid = torchvision.utils.make_grid(pred_map, nrow=num_images, normalize=False, padding=2)
        target_grid = torchvision.utils.make_grid(target_map, nrow=num_images, normalize=False, padding=2)
        
        writer.add_image(f"{prefix}/{map_name}_pred", pred_grid, global_step)
        writer.add_image(f"{prefix}/{map_name}_target", target_grid, global_step)


def log_input_renders(
    writer: SummaryWriter,
    input_renders: torch.Tensor,
    global_step: int,
    prefix: str = "val",
    max_images: int = 4
):
    """
    Log input multi-view renders to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        input_renders: Input tensor (B, num_views, 3, H, W)
        global_step: Current training step
        prefix: Prefix for tensorboard tags
        max_images: Maximum number of samples to show
    """
    batch_size, num_views = input_renders.shape[:2]
    num_images = min(batch_size, max_images)
    
    # Reshape to (B*num_views, 3, H, W) for grid
    views = input_renders[:num_images].reshape(-1, *input_renders.shape[2:])
    
    grid = torchvision.utils.make_grid(
        views,
        nrow=num_views,
        normalize=True,
        padding=2
    )
    
    writer.add_image(f"{prefix}/input_views", grid, global_step)


def visualize_pbr_batch(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    max_images: int = 4
) -> torch.Tensor:
    """
    Create a visualization of a batch (for saving or display).
    
    Args:
        pred: Dictionary of predicted PBR maps
        target: Dictionary of target PBR maps
        save_path: Optional path to save the visualization
        max_images: Maximum number of images to show
        
    Returns:
        Grid tensor
    """
    grid = create_comparison_grid(pred, target, max_images=max_images, include_error=True)
    
    if save_path:
        torchvision.utils.save_image(grid, save_path)
    
    return grid


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    pred = {
        "albedo": torch.rand(4, 3, 256, 256, device=device),
        "roughness": torch.rand(4, 1, 256, 256, device=device),
        "metallic": torch.rand(4, 1, 256, 256, device=device),
        "normal": F.normalize(torch.randn(4, 3, 256, 256, device=device), p=2, dim=1)
    }
    
    target = {
        "albedo": pred["albedo"] + 0.1 * torch.randn_like(pred["albedo"]),
        "roughness": pred["roughness"] + 0.1 * torch.randn_like(pred["roughness"]),
        "metallic": pred["metallic"] + 0.1 * torch.randn_like(pred["metallic"]),
        "normal": F.normalize(pred["normal"] + 0.1 * torch.randn_like(pred["normal"]), p=2, dim=1)
    }
    
    # Test comparison grid
    print("\n1. Creating comparison grid...")
    grid = create_comparison_grid(pred, target, max_images=2)
    print(f"   Grid shape: {grid.shape}")
    
    # Test error map
    print("\n2. Creating error map...")
    error = create_error_map(pred["albedo"], target["albedo"])
    print(f"   Error map shape: {error.shape}")
    
    # Test normal visualization
    print("\n3. Converting normal to RGB...")
    normal_rgb = normal_to_rgb(pred["normal"])
    print(f"   Normal RGB shape: {normal_rgb.shape}")
    print(f"   Normal RGB range: [{normal_rgb.min():.3f}, {normal_rgb.max():.3f}]")
    
    print("\n✓ Visualization module working correctly!")
