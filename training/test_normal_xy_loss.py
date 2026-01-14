#!/usr/bin/env python3
"""Test the normal XY magnitude loss to verify it provides useful gradients."""

import torch
import torch.nn.functional as F


def normal_xy_loss(pred_normal, target_normal):
    """
    Normal XY magnitude loss.
    
    Penalizes when predicted normals have less surface detail (smaller XY magnitude)
    than target normals. This directly fights the [0,0,1] collapse.
    """
    # Compute XY magnitude for pred and target
    pred_xy = (pred_normal[:, 0:2, :, :] ** 2).sum(dim=1, keepdim=True).sqrt()
    target_xy = (target_normal[:, 0:2, :, :] ** 2).sum(dim=1, keepdim=True).sqrt()
    
    # L1 loss on XY magnitude
    xy_loss = F.l1_loss(pred_xy, target_xy)
    
    # Additional penalty when pred XY is less than target XY (asymmetric)
    # This specifically penalizes "too flat" predictions
    xy_gap = F.relu(target_xy - pred_xy).mean()
    
    return xy_loss + xy_gap


def test_gradient_flow():
    """Test that gradients flow correctly and penalize flat predictions."""
    print("=" * 60)
    print("Testing Normal XY Magnitude Loss")
    print("=" * 60)
    
    # Create a realistic target normal map with surface detail
    # Normals point in various directions
    B, H, W = 2, 16, 16
    target = torch.randn(B, 3, H, W)
    target = F.normalize(target, dim=1)  # Unit normals
    
    # Case 1: Prediction is flat [0, 0, 1] (collapsed)
    print("\n[Case 1] Collapsed prediction (flat [0,0,1]):")
    pred_collapsed = torch.zeros(B, 3, H, W)
    pred_collapsed[:, 2, :, :] = 1.0  # Z = 1
    pred_collapsed.requires_grad_(True)
    
    loss_collapsed = normal_xy_loss(pred_collapsed, target)
    loss_collapsed.backward()
    
    print(f"  Target XY mag: {(target[:, 0:2, :, :] ** 2).sum(dim=1).sqrt().mean():.4f}")
    print(f"  Pred XY mag:   {(pred_collapsed[:, 0:2, :, :] ** 2).sum(dim=1).sqrt().mean():.4f}")
    print(f"  Loss:          {loss_collapsed.item():.4f}")
    print(f"  Gradient norm: {pred_collapsed.grad.norm().item():.4f}")
    print(f"  Grad X mean:   {pred_collapsed.grad[:, 0, :, :].mean().item():.6f}")
    print(f"  Grad Y mean:   {pred_collapsed.grad[:, 1, :, :].mean().item():.6f}")
    
    # Case 2: Prediction matches target well
    print("\n[Case 2] Good prediction (matches target):")
    pred_good = target.clone().detach()
    pred_good.requires_grad_(True)
    
    loss_good = normal_xy_loss(pred_good, target)
    loss_good.backward()
    
    print(f"  Pred XY mag:   {(pred_good[:, 0:2, :, :] ** 2).sum(dim=1).sqrt().mean():.4f}")
    print(f"  Loss:          {loss_good.item():.4f}")
    print(f"  Gradient norm: {pred_good.grad.norm().item():.4f}")
    
    # Case 3: Test with pre-tanh values (what the model actually outputs)
    print("\n[Case 3] Testing full tanh → normalize pipeline:")
    
    # Small pre-tanh values (what causes collapse)
    pre_tanh_small = torch.randn(B, 3, H, W) * 0.1  # Small values
    pre_tanh_small.requires_grad_(True)
    
    post_tanh_small = torch.tanh(pre_tanh_small)
    pred_small = F.normalize(post_tanh_small, dim=1)
    
    loss_small = normal_xy_loss(pred_small, target)
    loss_small.backward()
    
    print(f"  Pre-tanh std:  {pre_tanh_small.std().item():.4f}")
    print(f"  Post-tanh XY:  {(post_tanh_small[:, 0:2, :, :] ** 2).sum(dim=1).sqrt().mean():.4f}")
    print(f"  Pred XY mag:   {(pred_small[:, 0:2, :, :] ** 2).sum(dim=1).sqrt().mean():.4f}")
    print(f"  Pred Z mean:   {pred_small[:, 2, :, :].mean().item():.4f}")
    print(f"  Loss:          {loss_small.item():.4f}")
    print(f"  Gradient norm: {pre_tanh_small.grad.norm().item():.4f}")
    
    # Large pre-tanh values (healthy)
    pre_tanh_large = torch.randn(B, 3, H, W) * 1.0  # Larger values
    pre_tanh_large.requires_grad_(True)
    
    post_tanh_large = torch.tanh(pre_tanh_large)
    pred_large = F.normalize(post_tanh_large, dim=1)
    
    loss_large = normal_xy_loss(pred_large, target)
    loss_large.backward()
    
    print(f"\n  Pre-tanh std (large): {pre_tanh_large.std().item():.4f}")
    print(f"  Pred XY mag:   {(pred_large[:, 0:2, :, :] ** 2).sum(dim=1).sqrt().mean():.4f}")
    print(f"  Pred Z mean:   {pred_large[:, 2, :, :].mean().item():.4f}")
    print(f"  Loss:          {loss_large.item():.4f}")
    print(f"  Gradient norm: {pre_tanh_large.grad.norm().item():.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"  Collapsed (flat) loss: {loss_collapsed.item():.4f}")
    print(f"  Matched target loss:   {loss_good.item():.4f}")
    print(f"  Small pre-tanh loss:   {loss_small.item():.4f}")
    print(f"  Large pre-tanh loss:   {loss_large.item():.4f}")
    
    if loss_collapsed > loss_good and loss_small > loss_large:
        print("\n✓ Loss correctly penalizes flat/collapsed predictions!")
    else:
        print("\n✗ WARNING: Loss may not be working correctly!")
    
    return True


if __name__ == "__main__":
    test_gradient_flow()
