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


class WindowAttention(nn.Module):
    """
    Window-based spatial attention for PLK blocks.
    
    Unlike SE (channel) attention which applies the same weight to all pixels,
    window attention provides spatially-varying attention weights.
    This is crucial for normal maps where different regions need different treatment.
    
    Args:
        channels: Number of input channels
        window_size: Size of attention window (default 8)
        num_heads: Number of attention heads
    """
    
    def __init__(self, channels: int, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Relative position bias (learnable)
        self.relative_position_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        
        # Create position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        
        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, Hp, Wp = x.shape
        
        # Partition into windows: (B, C, H, W) -> (B*num_windows, C, ws, ws)
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, nH, nW, C, ws, ws)
        num_windows = (Hp // ws) * (Wp // ws)
        x = x.view(B * num_windows, C, ws, ws)
        
        # QKV
        qkv = self.qkv(x)  # (B*nW, 3C, ws, ws)
        qkv = qkv.view(B * num_windows, 3, self.num_heads, self.head_dim, ws * ws)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B*nW, heads, ws*ws, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B*nW, heads, ws*ws, ws*ws)
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias[
            self.relative_position_index.view(-1)
        ].view(ws * ws, ws * ws, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        out = attn @ v  # (B*nW, heads, ws*ws, head_dim)
        out = out.transpose(2, 3).contiguous().view(B * num_windows, C, ws, ws)
        
        # Project
        out = self.proj(out)
        
        # Reverse window partition
        out = out.view(B, Hp // ws, Wp // ws, C, ws, ws)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(B, C, Hp, Wp)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        return out


class ConvFFN(nn.Module):
    """
    Convolutional Feed-Forward Network from ESC.
    
    Structure: 1x1 expand → 3x3 depthwise → 1x1 project
    Provides local refinement before global PLK aggregation.
    """
    
    def __init__(self, channels: int, expansion: float = 2.0, use_bn: bool = True):
        super().__init__()
        hidden = int(channels * expansion)
        
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.norm = nn.BatchNorm2d(hidden) if use_bn else LayerNorm2d(hidden)
        self.project = nn.Conv2d(hidden, channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.expand(x))
        x = F.gelu(self.dwconv(x)) + x  # Residual inside FFN
        x = self.norm(x)
        x = self.project(x)
        return x


class ConvolutionalAttention(nn.Module):
    """
    Dynamic Convolutional Attention from ESC paper.
    
    Key insight: combines a content-adaptive dynamic 3x3 kernel with a static
    large kernel (PLK). The dynamic kernel allows the model to adapt its behavior
    based on the input content, which is crucial for complex outputs like normals.
    
    Structure:
        1. Generate dynamic 3x3 kernel from input features (via pooling + MLP)
        2. Apply dynamic depthwise conv (content-adaptive)
        3. Apply static PLK conv (global context)
        4. Sum the results
    
    The dynamic kernel generator is zero-initialized so it starts as identity,
    allowing gradual learning of content-adaptive behavior.
    
    Args:
        pdim: Number of channels to process with dynamic kernel
        proj_dim_in: Number of input channels for kernel generation (default: pdim)
    """
    
    def __init__(self, pdim: int, proj_dim_in: Optional[int] = None):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.sk_size = 3  # Dynamic kernel size (3x3)
        
        # Dynamic kernel generator: input features -> 3x3 depthwise kernel
        # Uses global pooling + MLP to generate per-sample kernels
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        
        # Zero-initialize the kernel generator output
        # This makes the dynamic kernel start as zero (identity behavior)
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)
    
    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic + static convolution.
        
        Args:
            x: Input features (B, C, H, W) where C >= pdim
            lk_filter: Static large kernel filter (pdim, 1, K, K)
        
        Returns:
            Output features (B, C, H, W)
        """
        bs = x.shape[0]
        
        # Split channels: first pdim channels get dynamic+static, rest pass through
        if x.shape[1] > self.pdim:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)
        else:
            x1 = x
            x2 = None
        
        # Generate dynamic kernel from input features
        # Shape: (B, pdim * 3 * 3) -> (B * pdim, 1, 3, 3)
        dynamic_kernel = self.dwc_proj(x[:, :self.proj_dim_in])
        dynamic_kernel = dynamic_kernel.reshape(bs * self.pdim, 1, self.sk_size, self.sk_size)
        
        # Apply dynamic depthwise conv
        # Reshape to (1, B*pdim, H, W) for grouped conv
        x1_reshaped = x1.reshape(1, bs * self.pdim, x1.shape[2], x1.shape[3])
        x1_dynamic = F.conv2d(
            x1_reshaped, 
            dynamic_kernel, 
            stride=1, 
            padding=self.sk_size // 2, 
            groups=bs * self.pdim
        )
        x1_dynamic = x1_dynamic.reshape(bs, self.pdim, x1.shape[2], x1.shape[3])
        
        # Apply static large kernel conv - REGULAR conv (paper's approach)
        # Filter is (pdim, pdim, K, K) for full channel mixing
        x1_static = F.conv2d(
            x1, 
            lk_filter, 
            stride=1, 
            padding=lk_filter.shape[-1] // 2
            # NO groups - regular conv for channel mixing
        )
        
        # Combine: static LK + dynamic adaptive
        x1_out = x1_static + x1_dynamic
        
        # Recombine with pass-through channels
        if x2 is not None:
            return torch.cat([x1_out, x2], dim=1)
        else:
            return x1_out
    
    def extra_repr(self):
        return f'pdim={self.pdim}, proj_dim_in={self.proj_dim_in}, dynamic_kernel_size={self.sk_size}'


class ConvAttnWrapper(nn.Module):
    """
    Wrapper that combines ConvolutionalAttention with aggregation.
    
    From ESC paper: applies PLK (dynamic + static) followed by 1x1 aggregation.
    """
    
    def __init__(self, dim: int, pdim: int, proj_dim_in: Optional[int] = None):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim, proj_dim_in)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
    
    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x = self.plk(x, lk_filter)
        x = self.aggr(x)
        return x


class PLKBlock(nn.Module):
    """
    Single PLK (Pre-computed Large Kernel) block from ESC.
    
    Structure matching paper:
        x = norm1(x); x = convffn(x)         # Local refinement (no residual)
        x = x + attn(norm2(x))               # Window attention with pre-norm
        for pconv, convffn in zip(...):
            x = x + pconv(convffn(x), plk)   # PLK+FFN pairs (no extra norm)
        x = conv_out(ln_out(x))
        return residual + gamma * x
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 17,
        use_bn: bool = True,
        use_window_attn: bool = True,
        window_size: int = 8,
        use_dynamic_kernel: bool = True,
        pdim: Optional[int] = None,
        conv_blocks: int = 5,
    ):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_window_attn = use_window_attn
        self.use_dynamic_kernel = use_dynamic_kernel
        self.conv_blocks = conv_blocks
        
        # pdim for partial conv - paper uses FIXED 16
        self.pdim = pdim if pdim is not None else 16
        
        # Paper uses LayerNorm throughout (not BatchNorm)
        self.norm1 = LayerNorm2d(channels)  # ln_proj
        
        # ConvFFN for local refinement (paper: proj)
        self.convffn = ConvFFN(channels, expansion=2.0)
        
        # Norm before attention (paper: ln_attn)
        self.norm2 = LayerNorm2d(channels)
        
        # Multiple PLK+FFN pairs
        if use_dynamic_kernel:
            self.pconvs = nn.ModuleList([
                ConvAttnWrapper(channels, self.pdim, proj_dim_in=self.pdim)
                for _ in range(conv_blocks)
            ])
        else:
            self.pconvs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                    LayerNorm2d(channels),
                    nn.GELU(),
                )
                for _ in range(conv_blocks)
            ])
        
        self.convffns = nn.ModuleList([
            ConvFFN(channels, expansion=1.25)
            for _ in range(conv_blocks)
        ])
        
        # Final 3x3 conv
        self.ln_out = LayerNorm2d(channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        # Attention mechanism
        if use_window_attn:
            num_heads = max(channels // 64, 4)
            self.attn = WindowAttention(channels, window_size=window_size, num_heads=num_heads)
        else:
            hidden = max(channels // 4, 32)
            self.attn = nn.Sequential(
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
        
        # 1. Initial projection (no residual)
        x = self.norm1(x)
        x = self.convffn(x)
        
        # 2. Window attention for spatial variation
        if self.use_window_attn:
            x = x + self.attn(self.norm2(x))
        else:
            B, C, _, _ = x.shape
            attn = self.attn(self.norm2(x)).view(B, C, 1, 1)
            x = x * attn
        
        # 3. Multiple PLK+FFN pairs (paper uses conv_blocks=5)
        for pconv, convffn in zip(self.pconvs, self.convffns):
            if self.use_dynamic_kernel:
                x = x + pconv(convffn(x), plk_filter)
            else:
                x_ffn = convffn(x)
                x_plk = F.conv2d(x_ffn, plk_filter, padding=self.padding, groups=self.channels)
                x = x + pconv(x_plk)
        
        # 4. Final 3x3 conv
        x = self.conv_out(self.ln_out(x))
        
        # Residual connection with scaling
        return residual + self.gamma * x


class PLKGenerator(nn.Module):
    """
    Generates the Pre-computed Large Kernel (PLK) filter.
    
    Paper's approach: Creates (pdim, pdim, K, K) kernel for REGULAR conv.
    Only the first pdim channels get processed with full channel mixing,
    rest of channels pass through unchanged.
    
    Optionally uses geometric ensemble to enforce rotational symmetry,
    which helps with directional outputs like normal maps.
    
    Args:
        pdim: Number of channels for the kernel (paper uses 16)
        kernel_size: Size of the large kernel (paper uses 13)
        use_geo_ensemble: If True, average kernel with rotated/flipped versions
    """
    
    def __init__(self, pdim: int = 16, kernel_size: int = 17, use_geo_ensemble: bool = True):
        super().__init__()
        
        self.pdim = pdim
        self.kernel_size = kernel_size
        self.use_geo_ensemble = use_geo_ensemble
        
        # Paper's approach: (pdim, pdim, K, K) for regular conv with channel mixing
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size))
        
        # Orthogonal initialization (from ESC paper)
        nn.init.orthogonal_(self.plk_filter)
    
    def _geo_ensemble(self, k: torch.Tensor) -> torch.Tensor:
        """
        Geometric ensemble: average kernel with all rotated/flipped versions.
        
        This enforces rotational symmetry, which is important for directional
        outputs like normal maps. From ESC paper.
        
        Args:
            k: Kernel tensor (pdim, pdim, K, K)
        
        Returns:
            Symmetrized kernel (pdim, pdim, K, K)
        """
        k_hflip = k.flip([3])
        k_vflip = k.flip([2])
        k_hvflip = k.flip([2, 3])
        k_rot90 = torch.rot90(k, -1, [2, 3])
        k_rot90_hflip = k_rot90.flip([3])
        k_rot90_vflip = k_rot90.flip([2])
        k_rot90_hvflip = k_rot90.flip([2, 3])
        k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
        return k
    
    def forward(self) -> torch.Tensor:
        """Generate the PLK filter."""
        plk = self.plk_filter
        
        # Apply geometric ensemble for rotational symmetry
        if self.use_geo_ensemble:
            plk = self._geo_ensemble(plk)
        
        return plk


class ConvAttnBottleneck(nn.Module):
    """
    ConvAttn bottleneck module using ESC-style PLK (Pre-computed Large Kernel).
    
    Structure matching the ESC paper:
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
        use_window_attn: Use window attention (True) or SE attention (False)
        window_size: Window size for window attention
        use_dynamic_kernel: Use dynamic+static PLK (True) or static-only (False)
    """
    
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 320,
        num_blocks: int = 3,
        kernel_size: int = 17,
        use_bn: bool = True,
        use_window_attn: bool = True,
        window_size: int = 8,
        expansion_ratio: int = 2,  # Legacy, kept for compatibility
        use_dynamic_kernel: bool = True  # ESC paper dynamic kernel
    ):
        super().__init__()
        
        self.bottleneck_channels = bottleneck_channels
        self.num_blocks = num_blocks
        self.use_dynamic_kernel = use_dynamic_kernel
        
        # pdim for PLK - paper uses FIXED 16 (partial conv on 16 channels only)
        # This allows large kernel + dynamic weights on only ~25% of features
        self.pdim = 16  # Fixed, not scaled
        
        # self.input_proj = nn.Sequential(
        #     nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(bottleneck_channels) if use_bn else LayerNorm2d(bottleneck_channels),
        #     nn.GELU()
        # )

        # Input already encoded by MobileNetV3, no projection needed
        # PLKBlock has its own norm1 at the start
        self.input_proj = nn.Identity()
        
        # Shared PLK generator - uses pdim for (pdim, pdim, K, K) filter
        self.plk_gen = PLKGenerator(pdim=self.pdim, kernel_size=kernel_size)
        
        # Stack of PLK blocks (all share the same PLK filter)
        self.blocks = nn.ModuleList([
            PLKBlock(
                channels=bottleneck_channels,
                kernel_size=kernel_size,
                use_bn=use_bn,
                use_window_attn=use_window_attn,
                window_size=window_size,
                use_dynamic_kernel=use_dynamic_kernel,
                pdim=self.pdim,  # Pass pdim to blocks
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
        use_window_attn: Use window attention (True) or SE attention (False)
        window_size: Window size for window attention
        use_dynamic_kernel: Use dynamic+static PLK (True) or static-only (False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 320,
        num_views: int = 4,
        num_blocks: int = 2,
        use_bn: bool = True,
        use_window_attn: bool = True,
        window_size: int = 8,
        use_dynamic_kernel: bool = True  # ESC paper dynamic kernel
    ):
        super().__init__()
        
        self.num_views = num_views
        
        # Squeeze dimension for lightweight dense refinement
        squeezed = max(in_channels // 4, 16)
        
        # Per-view feature refinement with REGULAR 3x3 conv (cross-channel mixing)
        # Lightweight: squeeze → regular 3x3 → expand
        self.view_refine = nn.ModuleList([
            nn.Sequential(
                # 1. Squeeze channels down
                nn.Conv2d(in_channels, squeezed, 1, bias=False),
                nn.GELU(),
                # 2. REGULAR 3x3 conv (NOT depthwise) - enables cross-channel mixing
                nn.Conv2d(squeezed, squeezed, 3, padding=1, bias=False),
                LayerNorm2d(squeezed),
                nn.GELU(),
                # 3. Expand back up
                nn.Conv2d(squeezed, in_channels, 1, bias=False)
            )
            for _ in range(num_views)
        ])
        
        # Fusion conv (concatenate views then project)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_views, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )
        
        # ConvAttn blocks for fused features
        self.bottleneck = ConvAttnBottleneck(
            in_channels=out_channels,
            bottleneck_channels=out_channels,
            num_blocks=num_blocks,
            use_bn=use_bn,
            use_window_attn=use_window_attn,
            window_size=window_size,
            use_dynamic_kernel=use_dynamic_kernel  # Enable dynamic kernel
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
