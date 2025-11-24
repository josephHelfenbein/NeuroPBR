import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal

'''
decoders are based off outputting 1024x1024 images
based on the fact that encoders are 1024x1024
'''

class ConvBlock(nn.Module):
    """double conv"""
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        # can switch norms to groupNorm or layerNorm if small epoch size ~1 or having stability issues

        # keeps output size the same, and doesn't recompute bias
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv = ConvBlock(out_channel + skip_channel, out_channel)

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


def _build_sr_2x():
    return nn.Sequential(
            # Increase capacity
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Prepare for PixelShuffle (4× channels)
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(inplace=True),

            # Super-resolve
            nn.PixelShuffle(upscale_factor=2),

            # Refine upsampled features
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


def _build_sr_4x():
    return nn.Sequential(
        # 2x
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.ReLU(inplace=True),

        nn.PixelShuffle(upscale_factor=2), # 256→512

        # 2x
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.ReLU(inplace=True),

        nn.PixelShuffle(upscale_factor=2),  # 512→1024

        # Refine
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )


class UNetDecoder(nn.Module):
    """
    sr_scale: 2 for stride=2 encoders (512→1024)
              0 for stride=1 encoders (1024→1024)
    """
    def __init__(self, in_channel: int, skip_channels: List[int], out_channel: int, sr_scale: Literal[0, 2, 4] = 2):
        super().__init__()

        self.decoders = nn.ModuleList()

        for skip_channel in skip_channels:
            out_ch = skip_channel if skip_channel <= 64 else skip_channel // 2
            self.decoders.append(
                DecoderBlock(in_channel, skip_channel, out_ch)
            )
            in_channel = out_ch

        # This should never run
        # safety net
        if in_channel != 64:
            self.decoders.append(
                nn.Conv2d(in_channel, 64, kernel_size=1)
            )

        if sr_scale == 2:
            # SR head (512→1024)
            self.sr_head = _build_sr_2x()

        elif sr_scale == 4:
            # SR head (256→1024) - kept for flexibility
            self.sr_head = _build_sr_4x()

        else:
            self.sr_head = nn.Identity()

        self.final_conv = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, x, skips):
        reversed_skips = skips[::-1]

        for dec, skip in zip(self.decoders, reversed_skips):
            x = dec(x, skip)

        # Super-resolve to 1024×1024
        x = self.sr_head(x)

        # Generate outputs
        x = self.final_conv(x)

        return x


class UNetDecoderHeads(nn.Module):
    def __init__(self, in_channel: int, skip_channels: List[int], out_channels: List[int], sr_scale: Literal[0, 2, 4] = 2):
        """shared decoder implementation"""
        super().__init__()
        self.decoders = nn.ModuleList()

        for skip_channel in skip_channels:
            out_ch = skip_channel if skip_channel <= 64 else skip_channel // 2
            self.decoders.append(
                DecoderBlock(in_channel, skip_channel, out_ch)
            )
            in_channel = out_ch

        if in_channel != 64:
            self.decoders.append(
                nn.Conv2d(in_channel, 64, kernel_size=1)
            )

        if sr_scale == 2:
            # SR head (512→1024)
            self.sr_head = _build_sr_2x()

        elif sr_scale == 4:
            # SR head (256→1024)
            self.sr_head = _build_sr_4x()

        else:
            self.sr_head = nn.Identity()

        self.heads = nn.ModuleList()

        for out_channel in out_channels:
            self.heads.append(
                nn.Conv2d(64, out_channel, kernel_size=1)
            )

    def forward(self, x, skips):
        reversed_skips = skips[::-1]

        for dec, skip in zip(self.decoders, reversed_skips):
            x = dec(x, skip)

        # Super-resolve to 1024×1024
        x = self.sr_head(x)

        outputs = [head(x) for head in self.heads]

        return outputs