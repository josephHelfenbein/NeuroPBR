"""
ConvAttn Student Generator for NeuroPBR.

Implements a student model using MobileNetV3 encoder with PLK (Pre-computed
Large Kernel) bottleneck for efficient long-range modeling. Replaces
transformer-based fusion with O(N) PLK blocks instead of O(N²) attention.

Features:
- MobileNetV3-Large encoder (pretrained, optionally frozen early layers)
- PLK bottleneck (2-4 blocks) with shared 17×17 learnable kernel
- SE-style channel attention for instance-adaptive weighting
- Support for feature-level distillation at bottleneck and decoder skips
- Much lower memory than ViT, enabling higher input resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import torch.utils.checkpoint as checkpoint

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_config import TrainConfig
from models.encoders.unet import UNetMobileNetV3Encoder
from models.decoders.unet import UNetDecoderHeads
from models.transformers.convattn import ConvAttnBottleneck, ConvAttnFusion, LayerNorm2d


class ConvAttnStudentGenerator(nn.Module):
    """
    Student generator using PLK ConvAttn for long-range modeling.
    
    Architecture:
        MobileNetV3 encoder → PLK bottleneck (2-4 blocks) →
        UNet decoder → 4 PBR map heads
    
    The PLK bottleneck replaces transformer-based fusion, providing:
    - Shared 17×17 learnable kernel across all blocks (pre-computed once)
    - SE-style channel attention for instance-adaptive weighting
    - O(N) memory vs O(N²) for ViT → enables higher resolution
    
    Args:
        config: Training configuration
        bottleneck_channels: Channels in PLK bottleneck (default 320)
        num_convattn_blocks: Number of PLK blocks (2-4)
    """
    
    def __init__(
        self,
        config: TrainConfig,
        bottleneck_channels: int = 320,
        num_convattn_blocks: int = 3,
        use_dynamic_kernel: bool = True
    ):
        super().__init__()
        self.config = config
        self.num_views = 3
        self.bottleneck_channels = bottleneck_channels
        
        # Build MobileNetV3 encoder
        self.encoder = self._build_encoder(config.model)
        self.latent_channels, self.encoder_skip_channels = self._inspect_encoder()
        
        # ConvAttn bottleneck for multi-view fusion
        # Input: encoder latent channels × num_views
        # Output: bottleneck_channels (to match decoder)
        self.fusion = ConvAttnFusion(
            in_channels=self.latent_channels,
            out_channels=bottleneck_channels,
            num_views=self.num_views,
            num_blocks=num_convattn_blocks,
            use_bn=True
        )
        
        # Project bottleneck to decoder expected channels if needed
        decoder_in_channels = bottleneck_channels
        skip_channels = self.encoder_skip_channels or config.model.decoder_skip_channels
        
        # Check if we need to project to match decoder expectations
        # The decoder expects the first skip channel to roughly match input
        if skip_channels and skip_channels[0] != bottleneck_channels:
            # Project to match decoder's expected input (usually 960 for MobileNetV3-Large)
            self.bottleneck_proj = nn.Sequential(
                nn.Conv2d(bottleneck_channels, skip_channels[0], kernel_size=1, bias=False),
                nn.BatchNorm2d(skip_channels[0]),
                nn.GELU()
            )
            decoder_in_channels = skip_channels[0]
        else:
            self.bottleneck_proj = nn.Identity()
            decoder_in_channels = bottleneck_channels
        
        # Decoder with 4 heads
        self.decoder = UNetDecoderHeads(
            in_channel=decoder_in_channels,
            skip_channels=skip_channels,
            out_channels=config.model.output_channels,
            sr_scale=config.model.decoder_sr_scale
        )
        
        # Store intermediate features for distillation
        self._bottleneck_features = None
        self._decoder_skip_features = []
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize new layers (encoder uses pretrained weights)."""
        for name, m in self.named_modules():
            if 'encoder' in name:
                continue  # Skip encoder (pretrained)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _build_encoder(self, model_config):
        """Build MobileNetV3 encoder."""
        return UNetMobileNetV3Encoder(
            in_channels=model_config.encoder_in_channels,
            backbone=model_config.encoder_backbone,
            freeze_backbone=model_config.freeze_backbone,
            freeze_bn=model_config.freeze_bn,
            stride=model_config.encoder_stride,
            skip=True
        )
    
    def _inspect_encoder(self) -> Tuple[int, List[int]]:
        """Determine encoder's latent and skip channel counts."""
        image_size = getattr(self.config.data, "image_size", (512, 512))
        in_ch = self.config.model.encoder_in_channels
        device = next(self.encoder.parameters()).device
        dtype = next(self.encoder.parameters()).dtype
        
        dummy = torch.zeros(1, in_ch, image_size[0], image_size[1], device=device, dtype=dtype)
        was_training = self.encoder.training
        self.encoder.eval()
        
        with torch.no_grad():
            latent, skips = self.encoder(dummy)
        
        if was_training:
            self.encoder.train()
        
        skip_channels = [s.shape[1] for s in skips] if skips else []
        skip_channels = list(reversed(skip_channels))
        
        return latent.shape[1], skip_channels
    
    def forward(
        self,
        views: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            views: (B, 3, 3, H, W) - batch of 3 RGB views
            return_features: If True, include intermediate features for distillation
            
        Returns:
            Dictionary with:
                - albedo, roughness, metallic, normal: PBR outputs
                - bottleneck_features (if return_features): For distillation
                - decoder_features (if return_features): Skip features for distillation
        """
        B, num_views, C, H, W = views.shape
        assert num_views == 3, f"Expected 3 views, got {num_views}"
        
        # Encode each view (shared weights)
        latents = []
        aggregated_skips = None
        
        for i in range(num_views):
            view = views[:, i]
            latent, skips = self.encoder(view)
            latents.append(latent)
            
            if aggregated_skips is None:
                aggregated_skips = [s.clone() for s in skips]
            else:
                aggregated_skips = [acc + s for acc, s in zip(aggregated_skips, skips)]
        
        # Average skip connections
        skips_list = [s / num_views for s in aggregated_skips] if aggregated_skips else []
        
        # Fuse views with ConvAttn
        fused = self.fusion(latents)
        
        # Store bottleneck features for distillation
        self._bottleneck_features = fused.clone()
        
        # Project to decoder input channels
        fused = self.bottleneck_proj(fused)
        
        # Decode to PBR maps
        outputs = self.decoder(fused, skips_list)
        
        # outputs: [albedo, roughness, metallic, normal]
        albedo, roughness, metallic, normal = outputs
        
        # Apply activations
        albedo = torch.sigmoid(albedo)
        roughness = torch.sigmoid(roughness)
        metallic = torch.sigmoid(metallic)
        normal = F.normalize(normal, p=2, dim=1)
        
        result = {
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "normal": normal
        }
        
        if return_features:
            result["bottleneck_features"] = self._bottleneck_features
            # Store two decoder skip features for distillation
            # These are the first two skip connections (highest resolution after bottleneck)
            if len(skips_list) >= 2:
                result["decoder_skip_0"] = skips_list[-1]  # Last (highest res) skip
                result["decoder_skip_1"] = skips_list[-2]  # Second last skip
        
        return result
    
    def get_encoder_params(self):
        """Get encoder parameters (for separate learning rate)."""
        return self.encoder.parameters()
    
    def get_non_encoder_params(self):
        """Get non-encoder parameters."""
        for name, param in self.named_parameters():
            if not name.startswith('encoder'):
                yield param


class ConvAttnDistillationLoss(nn.Module):
    """
    Distillation loss for ConvAttn student with feature matching.
    
    Combines:
    1. Output L1 loss (student vs teacher outputs)
    2. Feature L2 loss at bottleneck
    3. Feature L2 loss at two decoder skips
    
    Args:
        lambda_output: Weight for output L1 loss
        lambda_feat: Weight for feature losses (default 0.1)
        temperature: KL divergence temperature
        alpha: Distillation vs hard target balance
    """
    
    def __init__(
        self,
        lambda_output: float = 1.0,
        lambda_feat: float = 0.1,
        temperature: float = 4.0,
        alpha: float = 0.3
    ):
        super().__init__()
        self.lambda_output = lambda_output
        self.lambda_feat = lambda_feat
        self.temperature = temperature
        self.alpha = alpha
    
    def _l1_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute L1 loss across all PBR outputs."""
        loss = 0.0
        for key in ["albedo", "roughness", "metallic", "normal"]:
            if key in pred and key in target:
                pred_val = pred[key]
                target_val = target[key]
                
                # Resize if needed
                if pred_val.shape[-2:] != target_val.shape[-2:]:
                    target_val = F.interpolate(
                        target_val,
                        size=pred_val.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                loss = loss + F.l1_loss(pred_val, target_val)
        
        return loss / 4.0  # Average over 4 maps
    
    def _feature_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute L2 loss between feature maps with channel/spatial alignment."""
        B, C_s, H_s, W_s = student_feat.shape
        _, C_t, H_t, W_t = teacher_feat.shape
        
        # Resize spatial dimensions if needed
        if (H_s, W_s) != (H_t, W_t):
            teacher_feat = F.interpolate(
                teacher_feat,
                size=(H_s, W_s),
                mode='bilinear',
                align_corners=False
            )
        
        # Handle channel mismatch via 1D adaptive average pooling
        if C_s != C_t:
            # Reshape: (B, C_t, H, W) -> (B, H*W, C_t)
            teacher_flat = teacher_feat.permute(0, 2, 3, 1).reshape(B * H_s * W_s, 1, C_t)
            # Pool channels: (B*H*W, 1, C_t) -> (B*H*W, 1, C_s)
            teacher_pooled = F.adaptive_avg_pool1d(teacher_flat, C_s)
            # Reshape back: (B*H*W, 1, C_s) -> (B, H, W, C_s) -> (B, C_s, H, W)
            teacher_feat = teacher_pooled.reshape(B, H_s, W_s, C_s).permute(0, 3, 1, 2)
        
        return F.mse_loss(student_feat, teacher_feat)
    
    def forward(
        self,
        student_pred: Dict[str, torch.Tensor],
        teacher_pred: Dict[str, torch.Tensor],
        target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_pred: Student predictions with features
            teacher_pred: Teacher predictions with features
            target: Optional ground truth for hard loss
        
        Returns:
            total_loss: Combined loss
            loss_info: Dictionary with loss components
        """
        loss_info = {}
        total_loss = 0.0
        
        # Output L1 loss (student vs teacher)
        output_loss = self._l1_loss(student_pred, teacher_pred)
        total_loss = total_loss + self.lambda_output * output_loss
        loss_info["output_l1"] = output_loss.item()
        
        # Feature L2 at bottleneck
        if "bottleneck_features" in student_pred and "bottleneck_features" in teacher_pred:
            bottleneck_loss = self._feature_loss(
                student_pred["bottleneck_features"],
                teacher_pred["bottleneck_features"]
            )
            total_loss = total_loss + self.lambda_feat * bottleneck_loss
            loss_info["feat_bottleneck"] = bottleneck_loss.item()
        
        # Feature L2 at decoder skips
        for i in range(2):
            key = f"decoder_skip_{i}"
            if key in student_pred and key in teacher_pred:
                skip_loss = self._feature_loss(
                    student_pred[key],
                    teacher_pred[key]
                )
                total_loss = total_loss + self.lambda_feat * skip_loss
                loss_info[f"feat_skip_{i}"] = skip_loss.item()
        
        # Optional hard target loss
        if target is not None:
            hard_loss = self._l1_loss(student_pred, target)
            total_loss = total_loss + (1 - self.alpha) * hard_loss
            loss_info["hard_l1"] = hard_loss.item()
        
        loss_info["total"] = total_loss.item()
        
        return total_loss, loss_info


def create_convattn_student(
    config: TrainConfig,
    bottleneck_channels: int = 320,
    num_blocks: int = 3
) -> ConvAttnStudentGenerator:
    """
    Factory function to create ConvAttn student model.
    
    Args:
        config: Training configuration
        bottleneck_channels: ConvAttn bottleneck channels
        num_blocks: Number of ConvAttn blocks
    
    Returns:
        Configured ConvAttnStudentGenerator
    """
    # Ensure encoder is MobileNetV3
    if config.model.encoder_type != "mobilenetv3":
        print(f"Warning: Overriding encoder_type from {config.model.encoder_type} to mobilenetv3")
        config.model.encoder_type = "mobilenetv3"
    
    if "mobilenet" not in config.model.encoder_backbone:
        print(f"Warning: Overriding encoder_backbone to mobilenet_v3_large")
        config.model.encoder_backbone = "mobilenet_v3_large"
    
    return ConvAttnStudentGenerator(
        config=config,
        bottleneck_channels=bottleneck_channels,
        num_convattn_blocks=num_blocks,
        use_dynamic_kernel=True
    )
