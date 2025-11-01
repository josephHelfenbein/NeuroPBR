import torch
import torch.nn as nn
import os
from typing import Literal
from training.models.swinir.models.network_swinir import SwinIR

'''
models:
001_classicalSR_DF2K_s64w8_SwinIR‑M_x2.pth	Classical SR (×2 upscaling) trained on DF2K with patch size 64, window size 8. 
001_classicalSR_DF2K_s64w8_SwinIR‑M_x4.pth	Classical SR (×4 upscaling).
'''
class SwinIRWrapper(nn.Module):
    """
    Wrapper for SwinIR super-resolution model.
    """
    def __init__(
            self,
            pretrained_path: Literal[
                '001_classicalSR_DF2K_s64w8_SwinIR‑M_x2.pth',
                '001_classicalSR_DF2K_s64w8_SwinIR‑M_x3.pth',
                '001_classicalSR_DF2K_s64w8_SwinIR‑M_x4.pth',
                '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR‑M_x4_GAN.pth'
            ],
            upscale: Literal[2, 4] = 2,
            in_chans: int = 3,
            window_size: int = 8,
            upsampler: Literal['pixelshuffle', 'nearest+conv', 'none'] = 'pixelshuffle',
            resi_connection = '1conv',
            freeze_early_layers: bool = False,
            freeze_body: bool = False,
            freeze: bool = True, # full freeze (overrides freeze_early_layers)
            strict_load: bool = False
    ):
        super().__init__()

        self.model = SwinIR(
            upscale=upscale,
            in_chans=in_chans,
            window_size=window_size,
            mlp_ratio=2,
            upsampler=upsampler,
            resi_connection=resi_connection
        )

        if pretrained_path is not None:
            self._load_pretrained(os.path.join(os.getcwd(), '..', 'swinir', 'model_zoo', pretrained_path), strict=strict_load)

        if freeze:
            self.freeze()

        elif freeze_early_layers:
            self._freeze_early_layers(freeze_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _load_pretrained(self, pretrained_path: str, strict: bool = False) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained_path, map_location=device)

        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=strict)
        self.model.to(device)

    def _freeze_early_layers(self, freeze_body: bool = False) -> None:
        if hasattr(self.model, 'conv_first'):
            for param in self.model.conv_first.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'body') and freeze_body:
            for i, block in enumerate(self.model.body):
                if i < 2:
                    for param in block.parameters():
                        param.requires_grad = False

    def _freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
