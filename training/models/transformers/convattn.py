"""
ConvAttn Module for NeuroPBR Student Model.

Implements the ESC (Efficient Scalable Convolution) approach with PLK (Pre-computed
Large Kernel) filters. This is a simplified version optimized for already-encoded
features (from MobileNetV3).

The key insight from ESC is:
    1. Pre-compute a large kernel filter once (PLK)
    2. Pass it to each block for efficient long-range modeling
    3. Simple skip connection around the block stack

Structure:
    skip = feat
    plk_filter = self.plk_func(self.plk_filter)
    for block in self.blocks:
        feat = block(feat, plk_filter)
    feat = self.last(feat) + skip
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, List
import torch.utils.checkpoint as checkpoint


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (B, C, H, W)."""
    
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x


class PLKBlock(nn.Module):
    """
    Single PLK (Pre-computed Large Kernel) block from ESC.
    
    Takes a pre-computed large kernel filter and applies it efficiently.
    Uses depthwise separable structure for memory efficiency.
    
    Args:
        channels: Number of channels
        kernel_size: Size of the large kernel (default 17 for ~32×32 effective RF)
        use_bn: Use BatchNorm vs LayerNorm
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 17,
        use_bn: bool = True,
    ):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Pre-norm
        self.norm1 = nn.BatchNorm2d(channels) if use_bn else LayerNorm2d(channels)
        
        # Depthwise conv with the PLK filter
        # The actual filter weights come from the shared PLK
        self.dw_weight_shape = (channels, 1, kernel_size, kernel_size)
        
        # Pointwise mixing after PLK
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels) if use_bn else LayerNorm2d(channels),
            nn.GELU(),
        )
        
        # Channel attention (SE-style) for dynamic weighting
        hidden = max(channels // 4, 32)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        
        # Residual scaling
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor, plk_filter: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            plk_filter: Pre-computed large kernel (C, 1, K, K)
        
        Returns:
            Output features (B, C, H, W)
        """
        residual = x
        
        # Pre-norm
        x = self.norm1(x)
        
        # Apply PLK as depthwise conv
        x = F.conv2d(x, plk_filter, padding=self.padding, groups=self.channels)
        
        # Pointwise mixing
        x = self.pw(x)
        
        # Channel attention
        B, C, _, _ = x.shape
        attn = self.se(x).view(B, C, 1, 1)
        x = x * attn
        
        # Residual connection
        return residual + self.gamma * x


class PLKGenerator(nn.Module):
    """
    Generates the Pre-computed Large Kernel (PLK) filter.
    
    Creates a learnable large kernel that is shared across all blocks.
    The kernel is directly learnable with normalization applied.
    
    Args:
        channels: Number of channels for the kernel
        kernel_size: Size of the large kernel
    """
    
    def __init__(self, channels: int, kernel_size: int = 17):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Learnable kernel (depthwise format: C, 1, K, K)
        self.plk_filter = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size) * 0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with a Gaussian-like pattern centered in the kernel
        with torch.no_grad():
            center = self.kernel_size // 2
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                    value = torch.exp(torch.tensor(-dist / (self.kernel_size / 4)))
                    self.plk_filter[:, 0, i, j] = value * 0.1
    
    def forward(self) -> torch.Tensor:
        """Generate the normalized PLK filter."""
        # Normalize so each channel's kernel sums to 1 (like softmax attention)
        plk = self.plk_filter / (self.plk_filter.abs().sum(dim=(2, 3), keepdim=True) + 1e-6)
        return plk


class ConvAttnBottleneck(nn.Module):
    """
    ConvAttn bottleneck module using ESC-style PLK (Pre-computed Large Kernel).
    
    Simplified structure matching the ESC paper:
        skip = feat
        plk_filter = self.plk_func(self.plk_filter)
        for block in self.blocks:
            feat = block(feat, plk_filter)
        feat = self.last(feat) + skip
    
    Since our features are already encoded (from MobileNetV3), we skip the
    initial 3×3 encoder conv that ESC uses.
    
    Args:
        in_channels: Input channels from encoder
        bottleneck_channels: Internal bottleneck channels (default 320)
        num_blocks: Number of PLK blocks (2-4)
        kernel_size: PLK kernel size (default 17 for ~32×32 effective RF)
        use_bn: Use BatchNorm vs LayerNorm
    """
    
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 320,
        num_blocks: int = 3,
        kernel_size: int = 17,
        use_bn: bool = True,
        # Legacy params (ignored, kept for config compatibility)
        expansion_ratio: int = 2,
        use_dynamic_kernel: bool = True
    ):
        super().__init__()
        
        self.bottleneck_channels = bottleneck_channels
        self.num_blocks = num_blocks
        
        # Project to bottleneck dimension (replaces 3×3 encoder since we're already encoded)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels) if use_bn else LayerNorm2d(bottleneck_channels),
            nn.GELU()
        )
        
        # Shared PLK generator (creates the large kernel filter once)
        self.plk_gen = PLKGenerator(bottleneck_channels, kernel_size=kernel_size)
        
        # Stack of PLK blocks (all share the same PLK filter)
        self.blocks = nn.ModuleList([
            PLKBlock(
                channels=bottleneck_channels,
                kernel_size=kernel_size,
                use_bn=use_bn,
            )
            for _ in range(num_blocks)
        ])
        
        # Final projection (the "last" in ESC)
        self.last = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels) if use_bn else LayerNorm2d(bottleneck_channels),
        )
        
        # Output projection if needed (to match decoder expected channels)
        self.output_proj = nn.Identity()  # Will be set by caller if needed
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def set_output_channels(self, out_channels: int, use_bn: bool = True):
        """Set output projection to match decoder expected channels."""
        if out_channels != self.bottleneck_channels:
            self.output_proj = nn.Sequential(
                nn.Conv2d(self.bottleneck_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels) if use_bn else LayerNorm2d(out_channels),
                nn.GELU()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder output (B, C, H, W)
        
        Returns:
            Bottleneck features (B, bottleneck_channels, H, W)
        """
        # Project to bottleneck dimension
        feat = self.input_proj(x)
        
        # ESC pattern: skip + PLK blocks + last
        skip = feat
        
        # Generate PLK filter once (shared across all blocks)
        plk_filter = self.plk_gen()
        
        # Apply PLK blocks with shared filter
        for block in self.blocks:
            feat = block(feat, plk_filter)
        
        # Final projection + skip connection
        feat = self.last(feat) + skip
        
        # Output projection
        feat = self.output_proj(feat)
        
        return feat
    
    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate bottleneck features for distillation.
        Returns features after PLK blocks, before final projection.
        """
        feat = self.input_proj(x)
        plk_filter = self.plk_gen()
        
        for block in self.blocks:
            feat = block(feat, plk_filter)
        
        return feat


class GlobalContextMLP(nn.Module):
    """
    Optional global context module using pooling + MLP.
    
    Provides global context information that can be broadcast
    to all spatial locations. Lightweight alternative to global attention.
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        hidden_dim = max(channels // reduction, 64)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Global features
        global_feat = self.pool(x).view(B, C)
        global_feat = self.mlp(global_feat).view(B, C, 1, 1)
        
        # Add to input with learnable scaling
        return x + self.gamma * global_feat


class ConvAttnFusion(nn.Module):
    """
    ConvAttn-based multi-view fusion module.
    
    Replaces transformer-based multi-view fusion with efficient
    ConvAttn operations. Fuses multiple view features into a single
    representation.
    
    Args:
        in_channels: Channels per view
        out_channels: Output channels
        num_views: Number of input views
        num_blocks: Number of ConvAttn blocks after fusion
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 320,
        num_views: int = 4,
        num_blocks: int = 2,
        use_bn: bool = True
    ):
        super().__init__()
        
        self.num_views = num_views
        
        # Per-view feature refinement
        self.view_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels) if use_bn else LayerNorm2d(in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, in_channels, 1, bias=False)
            )
            for _ in range(num_views)
        ])
        
        # Fusion conv (concatenate views then project)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_views, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_bn else LayerNorm2d(out_channels),
            nn.GELU()
        )
        
        # ConvAttn blocks for fused features
        self.bottleneck = ConvAttnBottleneck(
            in_channels=out_channels,
            bottleneck_channels=out_channels,
            num_blocks=num_blocks,
            use_bn=use_bn
        )
    
    def forward(self, views: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            views: List of view features [(B, C, H, W), ...]
        
        Returns:
            Fused features (B, out_channels, H, W)
        """
        assert len(views) == self.num_views
        
        # Refine each view
        refined = [
            view + refine(view)
            for view, refine in zip(views, self.view_refine)
        ]
        
        # Concatenate and fuse
        concat = torch.cat(refined, dim=1)
        fused = self.fusion(concat)
        
        # Process with ConvAttn bottleneck
        out = self.bottleneck(fused)
        
        return out
