"""
Transformer and ConvAttn modules for NeuroPBR.

Uses ESC-style PLK (Pre-computed Large Kernel) for efficient long-range modeling.
"""

from .convattn import (
    # Core PLK components (ESC-style)
    PLKBlock,
    PLKGenerator,
    
    # Main modules
    ConvAttnBottleneck,
    ConvAttnFusion,
    GlobalContextMLP,
    
    # Utilities
    LayerNorm2d,
)

__all__ = [
    "PLKBlock",
    "PLKGenerator",
    "ConvAttnBottleneck",
    "ConvAttnFusion",
    "GlobalContextMLP",
    "LayerNorm2d",
]
