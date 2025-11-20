import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as ViTBlock # noqa
from einops import rearrange
from typing import Literal

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

    def forward(self, *latents):
        B, C, H, W = latents[0].shape # noqa

        assert C == self.dim, f"Input channels {C} doesn't match expected {self.dim}"

        tokens = [rearrange(l, 'b c h w -> b (h w) c') for l in latents]
        x = torch.cat(tokens, dim=1)  # [B, N*HW, C]

        for block in self.blocks:
            x = block(x)

        x = rearrange(x, 'b (v hw) c -> b hw (v c)', v=self.num_views)
        x = self.norm(self.fusion(x))

        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

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

    def forward(self, latent):
        B, C, H, W = latent.shape  # noqa

        assert C == self.dim, f"Input channels {C} doesn't match expected {self.dim}"

        x = rearrange(latent, 'b c h w -> b (h w) c')

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
