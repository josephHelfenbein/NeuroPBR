"""
Multi-view Fusion GAN Training Script for NeuroPBR

This script trains a model that:
1. Takes 3 rendered views as input (dirty renders with artifacts)
2. Uses multi-view fusion with Vision Transformer
3. Outputs 4 PBR maps (albedo, roughness, metallic, normal)
4. Trained with adversarial loss (GAN) + reconstruction losses

Dataset structure:
    root_dir/
    ├── input/
    │   ├── clean/sample_XXXX/{0,1,2}.png       (optional clean renders)
    │   ├── dirty/sample_XXXX/{0,1,2}.png       (dirty renders - training input)
    │   └── render_metadata.json                 (sample -> material mapping)
    └── output/
        └── material_name/
            ├── albedo.png                       (ground truth)
            ├── roughness.png
            ├── metallic.png
            └── normal.png

Usage:
    # Train with default config
    python train.py --data-root /path/to/data
    
    # Train with custom config
    python train.py --config configs/custom.py --data-root /path/to/data
    
    # Resume from checkpoint
    python train.py --resume checkpoints/model_epoch_50.pth
    
    # Distributed training (4 GPUs)
    torchrun --nproc_per_node=4 train.py --distributed --data-root /path/to/data
"""

import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Local imports
from train_config import TrainConfig, get_default_config, get_quick_test_config, get_lightweight_config
from models.encoders.unet import UNetEncoder, UNetStrideEncoder, UNetResNetEncoder
from models.decoders.unet import UNetDecoderHeads
from models.transformers.vision_transformer import ViTCrossViewFusion
from losses.losses import HybridLoss, PatchGANDiscriminator, discriminator_loss
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
        self.decoder = UNetDecoderHeads(
            in_channel=config.model.transformer_dim,
            skip_channels=config.model.decoder_skip_channels,
            out_channels=config.model.output_channels,
            sr_scale=config.model.decoder_sr_scale
        )
    
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
        skips_list = []
        
        for i in range(num_views):
            view = views[:, i]  # (B, C, H, W)
            latent, skips = self.encoder(view)
            latents.append(latent)
            if i == 0:
                skips_list = skips  # Use skips from first view (all should be similar)
        
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
    
    def __init__(self, config: TrainConfig, rank: int = 0):
        self.config = config
        self.rank = rank
        self.is_main_process = (rank == 0)
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        # Build models
        self.generator = MultiViewPBRGenerator(config).to(self.device)
        
        if config.model.use_gan:
            self.discriminator = PatchGANDiscriminator(
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
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Logging
        self.writer = None
        if self.is_main_process and config.training.use_tensorboard:
            log_dir = Path(config.training.checkpoint_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        # State
        self.current_epoch = config.training.start_epoch
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        print(f"[Rank {rank}] Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        if self.discriminator:
            print(f"  Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
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
        
        if scheduler_type == "cosine":
            g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.g_optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.optimizer.scheduler_min_lr
            )
            d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.d_optimizer,
                T_max=self.config.training.epochs,
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
        
        return g_sched, d_sched
    
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
                    
                    with autocast(enabled=self.config.training.use_amp):
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
            
            with autocast(enabled=self.config.training.use_amp):
                # Generate PBR maps
                pred_pbr = self.generator(input_renders)
                
                # Compute loss
                discriminator_for_loss = self.discriminator if use_gan else None
                g_loss, loss_info = self.criterion(
                    pred_pbr,
                    target,
                    discriminator=discriminator_for_loss
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
                    for key, val in loss_info.items():
                        self.writer.add_scalar(f"train/{key}", val, self.global_step)
            
            self.global_step += 1
        
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader) if use_gan else 0.0
        
        return avg_g_loss, avg_d_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int):
        """Validate the model."""
        self.generator.eval()
        if self.discriminator:
            self.discriminator.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        max_batches = self.config.training.val_batches or len(val_loader)
        
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
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if self.is_main_process:
            print(f"\n[Validation] Epoch {epoch} - Loss: {avg_loss:.4f}")
            if self.writer:
                self.writer.add_scalar("val/loss", avg_loss, epoch)
        
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
        
        # Save regular checkpoint
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        
        if self.discriminator and "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
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
            
            # Step schedulers
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
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    """Main training function."""
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
    
    # Override config with CLI args
    if args.data_root:
        config.data.data_root = args.data_root
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    
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
        root_dir=config.data.data_root,
        transform_mean=mean,
        transform_std=std,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        use_clean=config.data.use_clean_renders,
        split="train",
        val_ratio=config.data.val_ratio,
        image_size=config.data.image_size
    )
    
    val_loader = get_dataloader(
        root_dir=config.data.data_root,
        transform_mean=mean,
        transform_std=std,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=False,
        use_clean=config.data.use_clean_renders,
        split="val",
        val_ratio=config.data.val_ratio,
        image_size=config.data.image_size
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
    parser.add_argument("--data-root", type=str, default=None,
                      help="Root directory of dataset")
    parser.add_argument("--batch-size", type=int, default=None,
                      help="Batch size per GPU")
    
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
