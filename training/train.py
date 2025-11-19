"""
Multi-view Fusion GAN Training Script for NeuroPBR

This script trains a model that:
1. Takes 3 rendered views as input (clean by default, dirty optional)
2. Uses multi-view fusion with Vision Transformer
3. Outputs 4 PBR maps (albedo, roughness, metallic, normal)
4. Trained with adversarial loss (GAN) + reconstruction losses

Dataset structure:
    root_dir/
    ├── input/
    │   ├── clean/sample_XXXX/{0,1,2}.png       (default training input)
    │   ├── dirty/sample_XXXX/{0,1,2}.png       (optional dirty renders)
    │   └── render_metadata.json                 (sample -> material mapping)
    └── output/
        └── material_name/
            ├── albedo.png                       (ground truth)
            ├── roughness.png
            ├── metallic.png
            └── normal.png

Usage:
    # Train with default config
    python train.py --input-dir /path/to/data/input --output-dir /path/to/data/output
    
    # Train with custom config
    python train.py --config configs/custom.py --input-dir /path/to/data/input --output-dir /path/to/data/output
    
    # Force GPU/CPU selection
    python train.py --input-dir /path/to/data/input --output-dir /path/to/data/output --device cuda
    
    # Resume from checkpoint
    python train.py --resume checkpoints/model_epoch_50.pth
    
    # Distributed training (4 GPUs)
    torchrun --nproc_per_node=4 train.py --distributed --input-dir /path/to/data/input --output-dir /path/to/data/output
"""

import sys
import argparse
import random
import numpy as np
import inspect
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
except ImportError:  # Older PyTorch
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler

AUTOCAST_SUPPORTS_DEVICE = "device_type" in inspect.signature(_autocast).parameters
GRADSCALER_SUPPORTS_DEVICE = "device_type" in inspect.signature(_GradScaler.__init__).parameters

autocast = _autocast
GradScaler = _GradScaler

# Local imports
from train_config import TrainConfig, get_default_config, get_quick_test_config, get_lightweight_config
from models.encoders.unet import UNetEncoder, UNetStrideEncoder, UNetResNetEncoder
from models.decoders.unet import UNetDecoderHeads
from models.transformers.vision_transformer import ViTCrossViewFusion
from losses.losses import HybridLoss, discriminator_loss
from losses.losses import PatchGANDiscriminator as SimplePatchGANDiscriminator
from models.gan.discriminator import PatchGANDiscriminator as ConfigurablePatchGANDiscriminator
from utils.dataset import get_dataloader


class MultiViewPBRGenerator(nn.Module):
    """
    Multi-view fusion generator for PBR map prediction.
    
    Takes 3 rendered views, fuses them with ViT, outputs 4 PBR maps.
    """
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.num_views = 3
        
        # Create encoder (shared across all views)
        self.encoder = self._build_encoder(config.model)
        self.latent_channels, self.encoder_skip_channels = self._inspect_encoder()
        target_dim = config.model.transformer_dim
        if self.latent_channels == target_dim:
            self.latent_proj = nn.Identity()
        else:
            self.latent_proj = nn.Conv2d(self.latent_channels, target_dim, kernel_size=1, bias=False)
        
        # Cross-view fusion transformer
        if config.model.use_transformer:
            self.fusion = ViTCrossViewFusion(
                dim=config.model.transformer_dim,
                num_views=self.num_views,
                num_heads=config.model.transformer_num_heads,
                depth=config.model.transformer_depth,
                mlp_ratio=config.model.transformer_mlp_ratio,
                proj_drop=config.model.transformer_proj_drop,
                attn_drop=config.model.transformer_attn_drop,
                drop_path_rate=config.model.transformer_drop_path_rate
            )
        else:
            # Simple concatenation fusion
            self.fusion = nn.Conv2d(
                config.model.transformer_dim * self.num_views,
                config.model.transformer_dim,
                kernel_size=1
            )
        
        # Decoder with 4 heads (albedo, roughness, metallic, normal)
        skip_channels = self.encoder_skip_channels or config.model.decoder_skip_channels
        self.decoder = UNetDecoderHeads(
            in_channel=config.model.transformer_dim,
            skip_channels=skip_channels,
            out_channels=config.model.output_channels,
            sr_scale=config.model.decoder_sr_scale
        )
    
    def _inspect_encoder(self) -> Tuple[int, List[int]]:
        """Determine the encoder's latent and skip channel counts."""
        image_size = getattr(self.config.data, "image_size", (1024, 1024))
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

    def _build_encoder(self, model_config):
        """Build encoder based on config."""
        if model_config.encoder_type == "resnet":
            return UNetResNetEncoder(
                in_channels=model_config.encoder_in_channels,
                backbone=model_config.encoder_backbone,
                freeze_backbone=model_config.freeze_backbone,
                freeze_bn=model_config.freeze_bn,
                stride=model_config.encoder_stride,
                skip=True
            )
        elif model_config.encoder_type == "unet_stride":
            return UNetStrideEncoder(
                in_channels=model_config.encoder_in_channels,
                channel_list=model_config.encoder_channels,
                skip=True
            )
        else:  # "unet"
            return UNetEncoder(
                in_channels=model_config.encoder_in_channels,
                channel_list=model_config.encoder_channels,
                skip=True
            )
    
    def forward(self, views: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            views: (B, 3, 3, H, W) - batch of 3 RGB views
            
        Returns:
            Dictionary with keys: albedo, roughness, metallic, normal
        """
        B, num_views, C, H, W = views.shape
        assert num_views == 3, f"Expected 3 views, got {num_views}"
        
        # Encode each view separately (shared encoder weights)
        latents = []
        aggregated_skips = None
        
        for i in range(num_views):
            view = views[:, i]  # (B, C, H, W)
            latent, skips = self.encoder(view)
            latent = self.latent_proj(latent)
            latents.append(latent)
            if aggregated_skips is None:
                aggregated_skips = [s for s in skips]
            else:
                aggregated_skips = [acc + s for acc, s in zip(aggregated_skips, skips)]
        
        skips_list = [s / num_views for s in aggregated_skips] if aggregated_skips else []
        
        # Fuse latents with transformer
        if self.config.model.use_transformer:
            fused = self.fusion(*latents)
        else:
            # Simple concatenation + 1x1 conv
            fused = torch.cat(latents, dim=1)
            fused = self.fusion(fused)
        
        # Decode to PBR maps
        outputs = self.decoder(fused, skips_list)
        
        # outputs is a list: [albedo, roughness, metallic, normal]
        albedo, roughness, metallic, normal = outputs
        
        # Apply activations
        albedo = torch.sigmoid(albedo)  # [0, 1]
        roughness = torch.sigmoid(roughness)  # [0, 1]
        metallic = torch.sigmoid(metallic)  # [0, 1]
        normal = F.normalize(normal, p=2, dim=1)  # Normalized vector
        
        return {
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "normal": normal
        }


class Trainer:
    """Trainer class for multi-view PBR GAN."""
    
    def _resolve_device(self, preferred_device: Optional[str]) -> torch.device:
        """Resolve torch.device based on preference and availability."""
        preference = (preferred_device or "auto").lower()
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        auto_selected = preference == "auto"

        if auto_selected:
            if cuda_available:
                max_index = max(torch.cuda.device_count() - 1, 0)
                device_index = min(self.rank, max_index)
                return torch.device(f"cuda:{device_index}")
            if mps_available:
                return torch.device("mps")
            return torch.device("cpu")

        if preference.startswith("cuda"):
            if not cuda_available:
                raise RuntimeError("CUDA requested via --device but torch.cuda.is_available() is False.")
            return torch.device(preference)

        if preference.startswith("mps"):
            if not mps_available:
                raise RuntimeError("MPS requested via --device but torch.backends.mps.is_available() is False.")
            return torch.device("mps")

        if preference.startswith("cpu"):
            return torch.device("cpu")

        raise ValueError(
            f"Unknown device option '{preferred_device}'. Use 'auto', 'cuda', 'cuda:0', 'mps', or 'cpu'."
        )

    def __init__(self, config: TrainConfig, rank: int = 0):
        self.config = config
        self.rank = rank
        self.is_main_process = (rank == 0)

        # Set device
        self.device_preference = getattr(config.training, "device", "auto")
        self.device = self._resolve_device(self.device_preference)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        else:
            auto_selected = (self.device_preference is None) or (self.device_preference.lower() == "auto")
            if self.device.type == "cpu" and auto_selected:
                print("  No CUDA/MPS devices detected; training will run on CPU.")
        
        # Build models
        self.generator = MultiViewPBRGenerator(config).to(self.device)
        
        if config.model.use_gan:
            # Choose discriminator type based on config
            if config.model.discriminator_type == "configurable":
                self.discriminator = ConfigurablePatchGANDiscriminator(
                    in_channels=config.model.discriminator_in_channels,
                    n_filters=config.model.discriminator_ndf,
                    n_layers=config.model.discriminator_n_layers,
                    use_sigmoid=config.model.discriminator_use_sigmoid
                ).to(self.device)
            else:  # "simple"
                self.discriminator = SimplePatchGANDiscriminator(
                    in_channels=config.model.discriminator_in_channels,
                    ndf=config.model.discriminator_ndf
                ).to(self.device)
        else:
            self.discriminator = None
        
        # Build loss
        self.criterion = self._build_loss()
        
        # Build optimizers
        self.g_optimizer, self.d_optimizer = self._build_optimizers()
        
        # Build schedulers
        self.g_scheduler, self.d_scheduler = self._build_schedulers()
        
        # AMP scaler
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        scaler_kwargs = {}
        if self.use_amp and GRADSCALER_SUPPORTS_DEVICE:
            scaler_kwargs["device_type"] = self.amp_device_type
        self.scaler = GradScaler(**scaler_kwargs) if self.use_amp else None
        
        # Logging
        self.writer = None
        if self.is_main_process and config.training.use_tensorboard:
            log_dir = Path(config.training.checkpoint_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        # WandB logging
        self.use_wandb = config.training.use_wandb
        if self.use_wandb and self.is_main_process:
            try:
                import wandb  # type: ignore
                wandb.init(
                    project=config.training.wandb_project,
                    name=config.training.wandb_run_name,
                    config=config.to_dict()
                )
                print("  WandB logging enabled")
            except ImportError:
                print("  Warning: wandb not installed, disabling WandB logging")
                self.use_wandb = False
        
        # State
        self.current_epoch = config.training.start_epoch
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        print(f"[Rank {rank}] Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        if self.discriminator:
            print(f"  Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"  Mixed precision (AMP): {'enabled' if self.use_amp else 'disabled'}")
    
    def _build_loss(self):
        """Build loss function."""
        loss_config = {
            "w_l1": self.config.loss.w_l1,
            "w_ssim": self.config.loss.w_ssim,
            "w_normal": self.config.loss.w_normal,
            "w_perceptual": self.config.loss.w_perceptual,
            "w_gan": self.config.loss.w_gan,
            "w_albedo": self.config.loss.w_albedo,
            "w_roughness": self.config.loss.w_roughness,
            "w_metallic": self.config.loss.w_metallic,
            "w_normal_map": self.config.loss.w_normal_map,
            "use_perceptual": self.config.loss.use_perceptual,
            "gan_loss_type": self.config.loss.gan_loss_type
        }
        return HybridLoss(loss_config).to(self.device)
    
    def _build_optimizers(self):
        """Build optimizers for generator and discriminator."""
        # Generator optimizer
        if self.config.optimizer.g_optimizer == "adam":
            g_opt = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.config.optimizer.g_lr,
                betas=self.config.optimizer.g_betas
            )
        elif self.config.optimizer.g_optimizer == "adamw":
            g_opt = torch.optim.AdamW(
                self.generator.parameters(),
                lr=self.config.optimizer.g_lr,
                betas=self.config.optimizer.g_betas,
                weight_decay=self.config.optimizer.g_weight_decay
            )
        else:  # sgd
            g_opt = torch.optim.SGD(
                self.generator.parameters(),
                lr=self.config.optimizer.g_lr,
                momentum=0.9,
                weight_decay=self.config.optimizer.g_weight_decay
            )
        
        # Discriminator optimizer
        d_opt = None
        if self.discriminator:
            if self.config.optimizer.d_optimizer == "adam":
                d_opt = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=self.config.optimizer.d_lr,
                    betas=self.config.optimizer.d_betas
                )
            elif self.config.optimizer.d_optimizer == "adamw":
                d_opt = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=self.config.optimizer.d_lr,
                    betas=self.config.optimizer.d_betas,
                    weight_decay=self.config.optimizer.d_weight_decay
                )
        
        return g_opt, d_opt
    
    def _build_schedulers(self):
        """Build learning rate schedulers."""
        scheduler_type = self.config.optimizer.scheduler
        warmup_epochs = self.config.optimizer.scheduler_warmup_epochs
        
        # Base schedulers
        if scheduler_type == "cosine":
            g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.g_optimizer,
                T_max=self.config.training.epochs - warmup_epochs,
                eta_min=self.config.optimizer.scheduler_min_lr
            )
            d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.d_optimizer,
                T_max=self.config.training.epochs - warmup_epochs,
                eta_min=self.config.optimizer.scheduler_min_lr
            ) if self.d_optimizer else None
        
        elif scheduler_type == "step":
            g_sched = torch.optim.lr_scheduler.StepLR(
                self.g_optimizer,
                step_size=self.config.optimizer.step_size,
                gamma=self.config.optimizer.gamma
            )
            d_sched = torch.optim.lr_scheduler.StepLR(
                self.d_optimizer,
                step_size=self.config.optimizer.step_size,
                gamma=self.config.optimizer.gamma
            ) if self.d_optimizer else None
        
        elif scheduler_type == "plateau":
            g_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.g_optimizer,
                mode='min',
                patience=self.config.optimizer.patience,
                factor=self.config.optimizer.factor
            )
            d_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.d_optimizer,
                mode='min',
                patience=self.config.optimizer.patience,
                factor=self.config.optimizer.factor
            ) if self.d_optimizer else None
        
        else:  # "none"
            g_sched = None
            d_sched = None
        
        # Apply warmup if needed
        if warmup_epochs > 0 and g_sched is not None:
            # Simple linear warmup wrapper
            self.warmup_epochs = warmup_epochs
            self.warmup_start_lr = self.config.optimizer.g_lr / 10.0  # Start at 10% of target
        else:
            self.warmup_epochs = 0
        
        return g_sched, d_sched

    def _autocast(self):
        """Return the appropriate autocast context manager for current AMP setup."""
        if not self.use_amp:
            return nullcontext()
        if AUTOCAST_SUPPORTS_DEVICE:
            return autocast(device_type=self.amp_device_type)
        return autocast()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.generator.train()
        if self.discriminator:
            self.discriminator.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not self.is_main_process)
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        use_gan = self.config.model.use_gan and epoch >= self.config.training.gan_start_epoch
        
        for batch_idx, (input_renders, pbr_maps) in enumerate(pbar):
            # input_renders: (B, 3, 3, H, W) - 3 rendered views (dirty)
            # pbr_maps: (B, 4, 3, H, W) - 4 ground truth PBR maps
            
            input_renders = input_renders.to(self.device)
            pbr_maps = pbr_maps.to(self.device)
            
            # Prepare ground truth
            target = {
                "albedo": pbr_maps[:, 0],  # (B, 3, H, W)
                "roughness": pbr_maps[:, 1, 0:1],  # (B, 1, H, W) - take only R channel
                "metallic": pbr_maps[:, 2, 0:1],  # (B, 1, H, W)
                "normal": pbr_maps[:, 3]  # (B, 3, H, W)
            }
            
            # ==================== Train Discriminator ====================
            d_loss_val = 0.0
            if use_gan and self.discriminator:
                for _ in range(self.config.training.d_steps_per_g_step):
                    self.d_optimizer.zero_grad()
                    
                    with self._autocast():
                        # Generate fake PBR
                        with torch.no_grad():
                            fake_pbr = self.generator(input_renders)
                        
                        # Concatenate PBR maps for discriminator
                        real_concat = torch.cat([
                            target["albedo"],
                            target["roughness"],
                            target["metallic"],
                            target["normal"]
                        ], dim=1)  # (B, 8, H, W)
                        
                        fake_concat = torch.cat([
                            fake_pbr["albedo"].detach(),
                            fake_pbr["roughness"].detach(),
                            fake_pbr["metallic"].detach(),
                            fake_pbr["normal"].detach()
                        ], dim=1)
                        
                        # Discriminator predictions
                        real_logits = self.discriminator(real_concat)
                        fake_logits = self.discriminator(fake_concat)
                        
                        # Discriminator loss
                        d_loss = discriminator_loss(
                            real_logits,
                            fake_logits,
                            self.config.loss.gan_loss_type
                        ) * self.config.loss.w_discriminator
                    
                    # Backward
                    if self.scaler:
                        self.scaler.scale(d_loss).backward()
                        self.scaler.step(self.d_optimizer)
                        self.scaler.update()
                    else:
                        d_loss.backward()
                        self.d_optimizer.step()
                    
                    d_loss_val = d_loss.item()
            
            # ==================== Train Generator ====================
            self.g_optimizer.zero_grad()
            
            with self._autocast():
                # Generate PBR maps
                pred_pbr = self.generator(input_renders)
                
                # Compute loss (use albedo as RGB proxy for perceptual loss if enabled)
                discriminator_for_loss = self.discriminator if use_gan else None
                g_loss, loss_info = self.criterion(
                    pred_pbr,
                    target,
                    discriminator=discriminator_for_loss,
                    pred_rgb=pred_pbr["albedo"] if self.config.loss.use_perceptual else None,
                    target_rgb=target["albedo"] if self.config.loss.use_perceptual else None
                )
            
            # Backward
            if self.scaler:
                self.scaler.scale(g_loss).backward()
                if self.config.training.grad_clip_norm:
                    self.scaler.unscale_(self.g_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config.training.grad_clip_norm
                    )
                self.scaler.step(self.g_optimizer)
                self.scaler.update()
            else:
                g_loss.backward()
                if self.config.training.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config.training.grad_clip_norm
                    )
                self.g_optimizer.step()
            
            # Update metrics
            epoch_g_loss += loss_info["loss_total"]
            epoch_d_loss += d_loss_val
            
            # Logging
            if self.is_main_process and batch_idx % self.config.training.log_every_n_steps == 0:
                pbar.set_postfix({
                    "G_loss": f"{loss_info['loss_total']:.4f}",
                    "D_loss": f"{d_loss_val:.4f}" if use_gan else "N/A"
                })
                
                if self.writer:
                    self.writer.add_scalar("train/g_loss", loss_info["loss_total"], self.global_step)
                    self.writer.add_scalar("train/d_loss", d_loss_val, self.global_step)
                    self.writer.add_scalar("train/g_lr", self.g_optimizer.param_groups[0]['lr'], self.global_step)
                    for key, val in loss_info.items():
                        self.writer.add_scalar(f"train/{key}", val, self.global_step)
                
                if self.use_wandb:
                    import wandb  # type: ignore
                    log_dict = {
                        "train/g_loss": loss_info["loss_total"],
                        "train/d_loss": d_loss_val,
                        "train/g_lr": self.g_optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                        "step": self.global_step
                    }
                    for key, val in loss_info.items():
                        log_dict[f"train/{key}"] = val
                    wandb.log(log_dict, step=self.global_step)
            
            self.global_step += 1
        
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader) if use_gan else 0.0
        
        return avg_g_loss, avg_d_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int):
        """Validate the model."""
        from utils.metrics import compute_pbr_metrics
        from utils.visualization import log_images_to_tensorboard, log_input_renders
        
        self.generator.eval()
        if self.discriminator:
            self.discriminator.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = {}
        
        max_batches = self.config.training.val_batches or len(val_loader)
        
        if max_batches == 0:
            if self.is_main_process:
                print("\n[Validation] Skipping validation because no validation data is available.")
            return float("inf")
        
        # For image logging
        first_batch_logged = False
        
        for batch_idx, (input_renders, pbr_maps) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            
            input_renders = input_renders.to(self.device)
            pbr_maps = pbr_maps.to(self.device)
            
            target = {
                "albedo": pbr_maps[:, 0],
                "roughness": pbr_maps[:, 1, 0:1],
                "metallic": pbr_maps[:, 2, 0:1],
                "normal": pbr_maps[:, 3]
            }
            
            # Forward
            pred_pbr = self.generator(input_renders)
            
            # Loss (without GAN)
            _, loss_info = self.criterion(pred_pbr, target, discriminator=None)
            
            total_loss += loss_info["loss_total"]
            
            # Compute metrics
            batch_metrics = compute_pbr_metrics(pred_pbr, target, include_angular=True)
            for key, val in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(val)
            
            # Log images from first batch
            if not first_batch_logged and self.is_main_process:
                log_images = (
                    self.config.training.log_images_every_n_epochs > 0 and
                    epoch % self.config.training.log_images_every_n_epochs == 0
                )
                if log_images:
                    if self.writer:
                        log_input_renders(self.writer, input_renders, epoch, prefix="val")
                        log_images_to_tensorboard(self.writer, pred_pbr, target, epoch, prefix="val")
                    first_batch_logged = True
            
            num_batches += 1
        
        if num_batches == 0:
            if self.is_main_process:
                print("\n[Validation] No batches were processed; returning inf loss.")
            return float("inf")
        
        avg_loss = total_loss / num_batches
        
        # Average metrics
        avg_metrics = {key: sum(vals) / len(vals) for key, vals in all_metrics.items()}
        
        if self.is_main_process:
            print(f"\n[Validation] Epoch {epoch}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  PSNR: {avg_metrics.get('overall_psnr', 0):.2f} dB")
            print(f"  SSIM: {avg_metrics.get('overall_ssim', 0):.4f}")
            print(f"  Normal angle: {avg_metrics.get('normal_angle_mean', 0):.2f}°")
            
            if self.writer:
                self.writer.add_scalar("val/loss", avg_loss, epoch)
                for key, val in avg_metrics.items():
                    self.writer.add_scalar(f"val/{key}", val, epoch)
            
            if self.use_wandb:
                import wandb  # type: ignore
                log_dict = {"val/loss": avg_loss, "epoch": epoch}
                for key, val in avg_metrics.items():
                    log_dict[f"val/{key}"] = val
                wandb.log(log_dict, step=epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None, is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
        
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "generator_state_dict": self.generator.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
            "config": self.config,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss
        }
        
        if self.discriminator:
            checkpoint["discriminator_state_dict"] = self.discriminator.state_dict()
            checkpoint["d_optimizer_state_dict"] = self.d_optimizer.state_dict()
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save regular checkpoint (unless save_best_only is True)
        if not self.config.training.save_best_only:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Save latest
        latest_path = checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            with torch.serialization.safe_globals([TrainConfig]):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except pickle.UnpicklingError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        
        if self.discriminator and "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = min(checkpoint["epoch"] + 1, self.config.training.epochs)
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            # Apply warmup scheduler
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                lr = self.warmup_start_lr + (self.config.optimizer.g_lr - self.warmup_start_lr) * warmup_factor
                for param_group in self.g_optimizer.param_groups:
                    param_group['lr'] = lr
                if self.is_main_process:
                    print(f"Warmup: Epoch {epoch}, LR: {lr:.6f}")
            
            # Train
            avg_g_loss, avg_d_loss = self.train_epoch(train_loader, epoch)
            
            if self.is_main_process:
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Generator Loss: {avg_g_loss:.4f}")
                if self.discriminator:
                    print(f"  Discriminator Loss: {avg_d_loss:.4f}")
            
            # Validate
            val_loss = None
            if epoch % self.config.training.val_every_n_epochs == 0:
                val_loss = self.validate(val_loader, epoch)
                
                # Check if best
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Save checkpoint
                if epoch % self.config.training.save_every_n_epochs == 0:
                    self.save_checkpoint(epoch, val_loss, is_best)
            
            # Step schedulers (skip during warmup)
            if epoch >= self.warmup_epochs:
                if self.g_scheduler:
                    if isinstance(self.g_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_loss is not None:
                            self.g_scheduler.step(val_loss)
                    else:
                        self.g_scheduler.step()
                
                if self.d_scheduler:
                    if isinstance(self.d_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_loss is not None:
                            self.d_scheduler.step(val_loss)
                    else:
                        self.d_scheduler.step()
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    """Main training function."""
    # Check for distributed training
    if args.distributed:
        print("="*80)
        print("WARNING: Distributed training is not yet fully implemented!")
        print("The --distributed flag is parsed but DDP setup is not complete.")
        print("Multi-GPU training will fail. Please use single GPU for now.")
        print("="*80)
        print("\nTo implement distributed training, you need to:")
        print("1. Initialize torch.distributed with torch.distributed.init_process_group()")
        print("2. Wrap models with DistributedDataParallel")
        print("3. Use DistributedSampler for dataloaders")
        print("4. Properly handle rank/world_size for logging and checkpointing")
        print("="*80)
        import sys
        sys.exit(1)
    
    # Load config
    if args.config == "default":
        config = get_default_config()
    elif args.config == "quick_test":
        config = get_quick_test_config()
    elif args.config == "lightweight":
        config = get_lightweight_config()
    else:
        # Load custom config file
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        config = custom_config.get_config()
    
    prev_input_dir = config.data.input_dir
    prev_metadata_path = config.data.metadata_path
    prev_default_metadata = None
    if prev_input_dir:
        prev_default_metadata = str(Path(prev_input_dir) / "render_metadata.json")

    # Override config with CLI args
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.input_dir:
        config.data.input_dir = args.input_dir
    if args.output_dir:
        config.data.output_dir = args.output_dir
    metadata_explicit = False
    if args.metadata_path:
        config.data.metadata_path = args.metadata_path
        metadata_explicit = True
    if not metadata_explicit:
        input_changed = args.input_dir is not None and config.data.input_dir is not None
        prev_was_default = prev_default_metadata and prev_metadata_path == prev_default_metadata
        if input_changed or prev_was_default or config.data.metadata_path is None:
            if config.data.input_dir:
                config.data.metadata_path = str(Path(config.data.input_dir) / "render_metadata.json")
    if args.render_curriculum is not None:
        config.data.render_curriculum = args.render_curriculum
    elif args.use_dirty:
        config.data.render_curriculum = 2

    config.data.use_dirty_renders = (config.data.render_curriculum == 2)
    if args.device:
        config.training.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir

    if not config.data.input_dir:
        raise ValueError("Input directory is not set. Pass --input-dir or update config.data.input_dir")
    if not config.data.output_dir:
        raise ValueError("Output directory is not set. Pass --output-dir or update config.data.output_dir")
    if not config.data.metadata_path:
        raise ValueError("Metadata path is not set. Pass --metadata-path or keep render_metadata.json under the input directory")
    
    # Set seed
    set_seed(config.training.seed)
    
    # Get dataloaders
    print("Loading datasets...")
    
    # Prepare transform stats
    if config.transform.use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = config.transform.mean
        std = config.transform.std
    
    train_loader = get_dataloader(
        input_dir=config.data.input_dir,
        output_dir=config.data.output_dir,
        metadata_path=config.data.metadata_path,
        transform_mean=mean,
        transform_std=std,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        use_dirty=config.data.use_dirty_renders,
        curriculum_mode=config.data.render_curriculum,
        split="train",
        val_ratio=config.data.val_ratio,
        image_size=config.data.image_size,  # Use input size for now
        seed=config.training.seed
    )
    
    val_loader = get_dataloader(
        input_dir=config.data.input_dir,
        output_dir=config.data.output_dir,
        metadata_path=config.data.metadata_path,
        transform_mean=mean,
        transform_std=std,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=False,
        use_dirty=config.data.use_dirty_renders,
        curriculum_mode=config.data.render_curriculum,
        split="val",
        val_ratio=config.data.val_ratio,
        image_size=config.data.image_size,  # Use input size for now
        seed=config.training.seed
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = Trainer(config, rank=0)
    
    # Resume if specified
    if args.resume or config.training.resume_from:
        checkpoint_path = args.resume or config.training.resume_from
        trainer.load_checkpoint(checkpoint_path)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-View PBR GAN")
    
    # Config
    parser.add_argument("--config", type=str, default="default",
                      help="Config to use: 'default', 'quick_test', 'lightweight', or path to custom config")
    
    # Data
    parser.add_argument("--batch-size", type=int, default=None,
                      help="Batch size per GPU")
    parser.add_argument("--input-dir", type=str, default=None,
                      help="Directory containing rendered samples (expects dirty/ and optional clean/)")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory containing ground-truth material folders")
    parser.add_argument("--metadata-path", type=str, default=None,
                      help="Path to render_metadata.json mapping sample folders to materials")
    parser.add_argument("--use-dirty", action="store_true",
                      help="Use dirty renders instead of the default clean renders")
    parser.add_argument("--render-curriculum", type=int, choices=[0, 1, 2], default=None,
                      help="0=clean only, 1=match dataset clean/dirty ratio, 2=dirty only (overrides --use-dirty)")
    parser.add_argument("--device", type=str, default=None,
                      help="Device override: 'auto' (default), 'cuda', 'cuda:0', 'cpu', or 'mps'")
    
    # Training
    parser.add_argument("--epochs", type=int, default=None,
                      help="Number of epochs")
    parser.add_argument("--resume", type=str, default=None,
                      help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                      help="Directory to save checkpoints")
    
    # Distributed (for future)
    parser.add_argument("--distributed", action="store_true",
                      help="Use distributed training")
    
    args = parser.parse_args()
    
    main(args)
