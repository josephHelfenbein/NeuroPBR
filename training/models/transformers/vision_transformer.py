import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block as ViTBlock # noqa
from einops import rearrange
from typing import Literal

import torch.utils.checkpoint as checkpoint

# Base grid size for the learned spatial positional embedding.
# Interpolated at runtime to the actual latent (H, W), so the model
# stays size-agnostic across training configs (e.g. 64x64 or 128x128).
_POS_EMBED_BASE = 16


def _interpolate_pos_embed(pos_embed, H, W):
    """Bilinearly interpolate a learned (1, dim, base, base) spatial pos embed
    to (1, H*W, dim) tokens for the runtime latent size."""
    pos = F.interpolate(pos_embed, size=(H, W), mode='bilinear', align_corners=False)
    return rearrange(pos, 'b c h w -> b (h w) c')

class ViTCrossViewFusion(nn.Module):
    def __init__(
            self,
            dim=2048,
            num_views=3,
            num_heads: Literal[8, 16, 24, 28, 32] = 32,
            depth: Literal[2, 4, 6] = 4,
            mlp_ratio: Literal[2, 4] = 2,
            proj_drop: float = 0.2,
            attn_drop: float = 0.2,
            drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_views = num_views

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rates[i]
            )
            for i in range(depth)
        ])

        self.fusion = nn.Linear(dim * num_views, dim)
        self.norm = nn.LayerNorm(dim)

        # Learned 2D spatial positional embedding (size-agnostic via interpolation).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, dim, _POS_EMBED_BASE, _POS_EMBED_BASE)
        )
        # Learned per-view (segment) embedding, one vector per view.
        self.view_embed = nn.Parameter(torch.zeros(num_views, dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)

    def forward(self, *latents):
        B, C, H, W = latents[0].shape # noqa

        assert C == self.dim, f"Input channels {C} doesn't match expected {self.dim}"

        # Check for NaN in inputs
        for i, l in enumerate(latents):
            if torch.isnan(l).any():
                print(f"[ViT Warning] NaN in input latent {i}")

        # Spatial positional embedding interpolated to runtime (H, W).
        pos = _interpolate_pos_embed(self.pos_embed, H, W)  # [1, HW, C]

        tokens = []
        for i, l in enumerate(latents):
            t = rearrange(l, 'b c h w -> b (h w) c')
            # Add spatial position + per-view embedding before the transformer.
            t = t + pos + self.view_embed[i].view(1, 1, C)
            tokens.append(t)
        x = torch.cat(tokens, dim=1)  # [B, N*HW, C]

        for block in self.blocks:
            if self.training and x.requires_grad:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = rearrange(x, 'b (v hw) c -> b hw (v c)', v=self.num_views)
        x = self.norm(self.fusion(x))
        
        # Check for NaN in output
        output = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        if torch.isnan(output).any():
            print(f"[ViT Warning] NaN in output! Input had NaN: {any(torch.isnan(l).any() for l in latents)}")

        return output

class ViT(nn.Module):
    def __init__(
            self,
            dim=2048,
            num_heads: Literal[8, 16, 24, 28, 32] = 32,
            depth: Literal[2, 4, 6] = 4,
            mlp_ratio: Literal[2, 4] = 2,
            proj_drop: float = 0.2,
            attn_drop: float = 0.2,
            drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rates[i]
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        # Learned 2D spatial positional embedding (size-agnostic via interpolation).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, dim, _POS_EMBED_BASE, _POS_EMBED_BASE)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, latent):
        B, C, H, W = latent.shape  # noqa

        assert C == self.dim, f"Input channels {C} doesn't match expected {self.dim}"

        x = rearrange(latent, 'b c h w -> b (h w) c')
        x = x + _interpolate_pos_embed(self.pos_embed, H, W)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
