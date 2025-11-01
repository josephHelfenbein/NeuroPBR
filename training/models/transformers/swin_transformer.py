import torch
import torch.nn as nn
from timm.models.swin_transformer_v2 import SwinTransformerV2Block # noqa
from einops import rearrange
from typing import Tuple

class SwinCrossViewFusion(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim=2048, num_heads=16, depth=4, num_views=3, window_size=7):
        super().__init__()
        self.num_views = num_views

        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                drop_path = 0.1 * (i / depth) #increase if overfit, decrease if otherwise
            )
            for i in range(depth)
        ])

        self.fusion = nn.Linear(dim * num_views, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, *latents):
        B, C, H, W = latents[0].shape # noqa

        assert (H, W) == self.input_resolution, \
            f"Input size {(H, W)} doesn't match expected {self.input_resolution}"
        assert C == self.dim, \
            f"Input channels {C} doesn't match expected {self.dim}"

        tokens = [rearrange(l, 'b c h w -> b (h w) c') for l in latents]
        x = torch.cat(tokens, dim=1)  # [B, 3*H*W, C]

        for block in self.blocks:
            x = block(x)

        x = rearrange(x, 'b (v hw) c -> b hw (v c)', v=self.num_views) #tokenize
        x = self.norm(self.fusion(x))

        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W) # Back to spatial

class Swin(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim=2048, num_heads=16, depth=4, num_views=3, window_size=7):
        super().__init__()
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                drop_path = 0.1 * (i / depth) #increase if overfit, decrease if otherwise
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, latent):
        B, C, H, W = latent.shape # noqa

        assert (H, W) == self.input_resolution, \
            f"Input size {(H, W)} doesn't match expected {self.input_resolution}"
        assert C == self.dim, \
            f"Input channels {C} doesn't match expected {self.dim}"

        x = rearrange(latent, 'b c h w -> b (h w) c')

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)