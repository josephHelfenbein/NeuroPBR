import sys
import argparse
import random
import numpy as np
import pickle
import inspect
from pathlib import Path
from typing import Dict, Tuple, Optional
from contextlib import nullcontext
from tqdm import tqdm

# Add parent directory to path for imports
# Insert at the beginning to ensure 'train' resolves to the parent module, not this script
sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Import components from parent directory
from train import MultiViewPBRGenerator
from train_config import TrainConfig, get_default_config
from models.encoders.unet import UNetMobileNetV3Encoder
from models.decoders.unet import UNetDecoderHeads
from models.transformers.vision_transformer import ViTCrossViewFusion
from losses.losses import HybridLoss
from utils.dataset import get_dataloader


class StudentGenerator(MultiViewPBRGenerator):
    """
    Student generator that extends teacher generator to support MobileNetV3.

    The student model uses the same architecture as the teacher (multi-view fusion)
    but with a lightweight MobileNetV3 encoder instead of ResNet.
    """

    def _build_encoder(self, model_config):
        """Build MobileNetV3 encoder for student model."""
        if model_config.encoder_type == "mobilenetv3":
            return UNetMobileNetV3Encoder(
                in_channels=model_config.encoder_in_channels,
                backbone=model_config.encoder_backbone,
                freeze_backbone=model_config.freeze_backbone,
                freeze_bn=model_config.freeze_bn,
                stride=model_config.encoder_stride,
                skip=True
            )
        else:
            # Fall back to parent implementation for other encoder types
            return super()._build_encoder(model_config)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for PBR maps.

    Combines:
    1. Hard target loss: Standard reconstruction loss on ground truth
    2. Soft target loss: KL divergence between student and teacher predictions

    The soft targets use temperature scaling to create smoother probability distributions,
    which helps the student learn the similarity structure in the data.

    Reference: https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        hard_loss_config: Optional[Dict] = None
    ):
        """
        Args:
            temperature: Temperature for softening probability distributions (T).
                        Higher T = softer distributions. Typically 2-10.
            alpha: Weight for distillation loss. Final loss = alpha * soft + (1-alpha) * hard
                   Typically 0.1-0.5
            hard_loss_config: Configuration dict for HybridLoss (reconstruction loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

        # Hard target loss (standard reconstruction)
        if hard_loss_config is None:
            hard_loss_config = {
                "w_l1": 1.0,
                "w_ssim": 0.3,
                "w_normal": 0.5,
                "w_gan": 0.0,  # No GAN for student training
                "w_albedo": 1.0,
                "w_roughness": 1.0,
                "w_metallic": 1.0,
                "w_normal_map": 1.0,
                "gan_loss_type": "hinge"
            }
        self.hard_loss = HybridLoss(hard_loss_config)

    def _apply_temperature(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Apply temperature scaling to create softer probability distributions.

        For values already in [0,1] (like sigmoid outputs), we apply temperature
        to their logits to get smoother distributions.
        """
        # Convert from probability space to logit space
        eps = 1e-7
        x_clamped = torch.clamp(x, eps, 1 - eps)
        logits = torch.log(x_clamped / (1 - x_clamped))

        # Apply temperature scaling
        soft_probs = torch.sigmoid(logits / temperature)
        return soft_probs

    def _kl_divergence_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Compute KL divergence between student and teacher outputs.

        For continuous values (PBR maps), we treat each pixel as a Bernoulli distribution
        and compute KL divergence. The temperature scaling makes distributions softer.

        KL(P||Q) = P * log(P/Q) + (1-P) * log((1-P)/(1-Q))

        Returns loss scaled by T^2 to account for gradient magnitude.
        """
        # Apply temperature scaling
        student_soft = self._apply_temperature(student_output, temperature)
        teacher_soft = self._apply_temperature(teacher_output, temperature)

        # Compute KL divergence for Bernoulli distributions
        eps = 1e-7
        student_soft = torch.clamp(student_soft, eps, 1 - eps)
        teacher_soft = torch.clamp(teacher_soft, eps, 1 - eps)

        kl_loss = teacher_soft * torch.log(teacher_soft / student_soft) + \
                  (1 - teacher_soft) * torch.log((1 - teacher_soft) / (1 - student_soft))

        # Scale by T^2 to account for gradient magnitude (standard in distillation)
        kl_loss = kl_loss.mean() * (temperature ** 2)

        return kl_loss

    def forward(
        self,
        student_pred: Dict[str, torch.Tensor],
        teacher_pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            student_pred: Student predictions {albedo, roughness, metallic, normal}
            teacher_pred: Teacher predictions (same keys)
            target: Ground truth PBR maps (same keys)

        Returns:
            total_loss: Combined weighted loss
            loss_info: Dictionary with individual loss components
        """
        # Hard target loss (student vs ground truth)
        hard_loss, hard_loss_info = self.hard_loss(
            student_pred, target, discriminator=None
        )

        # Soft target loss (student vs teacher)
        # Compute KL divergence for each PBR map
        soft_loss = 0.0
        soft_loss_info = {}

        # Albedo (3 channels)
        albedo_kl = self._kl_divergence_loss(
            student_pred["albedo"],
            teacher_pred["albedo"],
            self.temperature
        )
        soft_loss += albedo_kl
        soft_loss_info["distill_albedo"] = albedo_kl.item()

        # Roughness (1 channel)
        roughness_kl = self._kl_divergence_loss(
            student_pred["roughness"],
            teacher_pred["roughness"],
            self.temperature
        )
        soft_loss += roughness_kl
        soft_loss_info["distill_roughness"] = roughness_kl.item()

        # Metallic (1 channel)
        metallic_kl = self._kl_divergence_loss(
            student_pred["metallic"],
            teacher_pred["metallic"],
            self.temperature
        )
        soft_loss += metallic_kl
        soft_loss_info["distill_metallic"] = metallic_kl.item()

        # Normal (3 channels)
        # For normals, use MSE instead of KL since they're already normalized vectors
        normal_mse = F.mse_loss(student_pred["normal"], teacher_pred["normal"])
        soft_loss += normal_mse * (self.temperature ** 2)  # Scale for consistency
        soft_loss_info["distill_normal"] = normal_mse.item()

        soft_loss_info["distill_total"] = soft_loss.item()

        # Combine hard and soft losses
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        # Merge loss info
        loss_info = {
            "loss_total": total_loss.item(),
            "loss_hard": hard_loss.item(),
            "loss_soft": soft_loss.item(),
            **hard_loss_info,
            **soft_loss_info
        }

        return total_loss, loss_info


class Trainer:
    """Trainer for student model with knowledge distillation."""

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

    def __init__(
        self,
        config: TrainConfig,
        teacher_checkpoint_path: Optional[str] = None,
        temperature: float = 4.0,
        alpha: float = 0.3,
        rank: int = 0
    ):
        self.config = config
        self.rank = rank
        self.is_main_process = (rank == 0)
        self.temperature = temperature
        self.alpha = alpha

        # Set device
        self.device_preference = getattr(config.training, "device", "auto")
        self.device = self._resolve_device(self.device_preference)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        else:
            auto_selected = (self.device_preference is None) or (self.device_preference.lower() == "auto")
            if self.device.type == "cpu" and auto_selected:
                print("  No CUDA/MPS devices detected; training will run on CPU.")

        # Build student model (MobileNetV3-based)
        if self.is_main_process:
            print(f"Building student model ({config.model.encoder_type})...")
        self.student = StudentGenerator(config).to(self.device)

        # Load and freeze teacher model (for inference only during training)
        if teacher_checkpoint_path:
            if self.is_main_process:
                print(f"Loading teacher model from {teacher_checkpoint_path}...")
            self.teacher = self._load_teacher(teacher_checkpoint_path)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher = None
            if self.is_main_process:
                print("No teacher checkpoint provided. Assuming shards contain teacher predictions.")

        # Build distillation loss
        hard_loss_config = {
            "w_l1": config.loss.w_l1,
            "w_ssim": config.loss.w_ssim,
            "w_normal": config.loss.w_normal,
            "w_gan": 0.0,  # No GAN for student
            "w_albedo": config.loss.w_albedo,
            "w_roughness": config.loss.w_roughness,
            "w_metallic": config.loss.w_metallic,
            "w_normal_map": config.loss.w_normal_map,
            "gan_loss_type": config.loss.gan_loss_type
        }
        self.criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            hard_loss_config=hard_loss_config
        ).to(self.device)

        # Build optimizer (only for student)
        self.optimizer = self._build_optimizer()

        # Build scheduler
        self.scheduler = self._build_scheduler()

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
                import wandb
                wandb.init(
                    project=config.training.wandb_project,
                    name=f"{config.training.wandb_run_name}_student" if config.training.wandb_run_name else "student",
                    config={
                        **config.to_dict(),
                        "distillation_temperature": temperature,
                        "distillation_alpha": alpha
                    }
                )
                print("  WandB logging enabled")
            except ImportError:
                print("  Warning: wandb not installed, disabling WandB logging")
                self.use_wandb = False

        # State
        self.current_epoch = config.training.start_epoch
        self.global_step = 0
        self.best_val_loss = float('inf')

        if self.is_main_process:
            print(f"[Rank {rank}] Student Trainer initialized")
            print(f"  Device: {self.device}")
            print(f"  Student params: {sum(p.numel() for p in self.student.parameters()):,}")
            if self.teacher:
                print(f"  Teacher params: {sum(p.numel() for p in self.teacher.parameters()):,}")
            else:
                print("  Teacher params: N/A (using pre-computed shards)")
            print(f"  Distillation temperature: {temperature}")
            print(f"  Distillation alpha: {alpha}")
            print(f"  Mixed precision (AMP): {'enabled' if self.use_amp else 'disabled'}")

    def _load_teacher(self, checkpoint_path: str) -> nn.Module:
        """Load pre-trained teacher model (frozen, for inference only)."""
        try:
            with torch.serialization.safe_globals([TrainConfig]):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except pickle.UnpicklingError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Get teacher config
        teacher_config = checkpoint.get("config", get_default_config())

        # Build teacher model using original MultiViewPBRGenerator
        teacher = MultiViewPBRGenerator(teacher_config).to(self.device)
        teacher.load_state_dict(checkpoint["generator_state_dict"])

        return teacher

    def _build_optimizer(self):
        """Build optimizer for student."""
        if self.config.optimizer.g_optimizer == "adam":
            return torch.optim.Adam(
                self.student.parameters(),
                lr=self.config.optimizer.g_lr,
                betas=self.config.optimizer.g_betas
            )
        elif self.config.optimizer.g_optimizer == "adamw":
            return torch.optim.AdamW(
                self.student.parameters(),
                lr=self.config.optimizer.g_lr,
                betas=self.config.optimizer.g_betas,
                weight_decay=self.config.optimizer.g_weight_decay
            )
        else:  # sgd
            return torch.optim.SGD(
                self.student.parameters(),
                lr=self.config.optimizer.g_lr,
                momentum=0.9,
                weight_decay=self.config.optimizer.g_weight_decay
            )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler_type = self.config.optimizer.scheduler
        warmup_epochs = self.config.optimizer.scheduler_warmup_epochs

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs - warmup_epochs,
                eta_min=self.config.optimizer.scheduler_min_lr
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.optimizer.step_size,
                gamma=self.config.optimizer.gamma
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.optimizer.patience,
                factor=self.config.optimizer.factor
            )
        else:  # "none"
            scheduler = None

        # Apply warmup if needed
        if warmup_epochs > 0 and scheduler is not None:
            self.warmup_epochs = warmup_epochs
            self.warmup_start_lr = self.config.optimizer.g_lr / 10.0
        else:
            self.warmup_epochs = 0

        return scheduler

    def _autocast(self):
        """Return the appropriate autocast context manager for current AMP setup."""
        if not self.use_amp:
            return nullcontext()
        if AUTOCAST_SUPPORTS_DEVICE:
            return autocast(device_type=self.amp_device_type)
        return autocast()

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch with distillation."""
        self.student.train()
        if self.teacher:
            self.teacher.eval()  # Teacher always in eval mode (inference only)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not self.is_main_process)

        epoch_loss = 0.0

        for batch_idx, batch_data in enumerate(pbar):
            # Handle both dataset types
            if len(batch_data) == 3:
                input_renders, pbr_maps, teacher_pred_batch = batch_data
                # teacher_pred_batch is a dict of tensors. Move to device.
                teacher_pred = {k: v.to(self.device) for k, v in teacher_pred_batch.items()}
            else:
                input_renders, pbr_maps = batch_data
                teacher_pred = None

            input_renders = input_renders.to(self.device)
            pbr_maps = pbr_maps.to(self.device)

            # Prepare ground truth
            target = {
                "albedo": pbr_maps[:, 0],
                "roughness": pbr_maps[:, 1, 0:1],
                "metallic": pbr_maps[:, 2, 0:1],
                "normal": pbr_maps[:, 3]
            }

            # Get teacher predictions (no gradient - inference only)
            if teacher_pred is None:
                if self.teacher is None:
                     raise RuntimeError("Teacher model not loaded and no teacher predictions in batch!")
                with torch.no_grad():
                    teacher_pred = self.teacher(input_renders)

            # Train student
            self.optimizer.zero_grad()

            with self._autocast():
                # Student forward
                student_pred = self.student(input_renders)

                # Distillation loss
                loss, loss_info = self.criterion(student_pred, teacher_pred, target)

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.config.training.grad_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.config.training.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.config.training.grad_clip_norm
                    )
                self.optimizer.step()

            # Update metrics
            epoch_loss += loss_info["loss_total"]

            # Logging
            if self.is_main_process and batch_idx % self.config.training.log_every_n_steps == 0:
                pbar.set_postfix({
                    "loss": f"{loss_info['loss_total']:.4f}",
                    "hard": f"{loss_info['loss_hard']:.4f}",
                    "soft": f"{loss_info['loss_soft']:.4f}"
                })

                if self.writer:
                    self.writer.add_scalar("train/loss_total", loss_info["loss_total"], self.global_step)
                    self.writer.add_scalar("train/loss_hard", loss_info["loss_hard"], self.global_step)
                    self.writer.add_scalar("train/loss_soft", loss_info["loss_soft"], self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    for key, val in loss_info.items():
                        if key not in ["loss_total", "loss_hard", "loss_soft"]:
                            self.writer.add_scalar(f"train/{key}", val, self.global_step)

                if self.use_wandb:
                    import wandb
                    log_dict = {
                        "train/loss_total": loss_info["loss_total"],
                        "train/loss_hard": loss_info["loss_hard"],
                        "train/loss_soft": loss_info["loss_soft"],
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                        "step": self.global_step
                    }
                    for key, val in loss_info.items():
                        if key not in ["loss_total", "loss_hard", "loss_soft"]:
                            log_dict[f"train/{key}"] = val
                    wandb.log(log_dict, step=self.global_step)

            self.global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int):
        """Validate the student model."""
        from utils.metrics import compute_pbr_metrics
        from utils.visualization import log_images_to_tensorboard, log_input_renders

        self.student.eval()
        if self.teacher:
            self.teacher.eval()

        total_loss = 0.0
        num_batches = 0
        all_metrics = {}

        max_batches = self.config.training.val_batches or len(val_loader)

        if max_batches == 0:
            if self.is_main_process:
                print("\n[Validation] Skipping validation because no validation data is available.")
            return float("inf")

        first_batch_logged = False

        for batch_idx, batch_data in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            # Handle both dataset types
            if len(batch_data) == 3:
                input_renders, pbr_maps, teacher_pred_batch = batch_data
                teacher_pred = {k: v.to(self.device) for k, v in teacher_pred_batch.items()}
            else:
                input_renders, pbr_maps = batch_data
                teacher_pred = None

            input_renders = input_renders.to(self.device)
            pbr_maps = pbr_maps.to(self.device)

            target = {
                "albedo": pbr_maps[:, 0],
                "roughness": pbr_maps[:, 1, 0:1],
                "metallic": pbr_maps[:, 2, 0:1],
                "normal": pbr_maps[:, 3]
            }

            # Get predictions
            if teacher_pred is None:
                if self.teacher is None:
                     # If validating on a dataset without teacher preds and no teacher model, 
                     # we can't compute distillation loss. 
                     # But we can still compute metrics against GT.
                     # However, the loss function expects teacher_pred.
                     # Let's assume we skip soft loss or fail.
                     # For now, let's fail to be safe.
                     raise RuntimeError("Teacher model not loaded and no teacher predictions in batch!")
                teacher_pred = self.teacher(input_renders)
            
            student_pred = self.student(input_renders)  # Student runs independently

            # Compute loss
            _, loss_info = self.criterion(student_pred, teacher_pred, target)
            total_loss += loss_info["loss_total"]

            # Compute metrics (student vs ground truth)
            batch_metrics = compute_pbr_metrics(student_pred, target, include_angular=True)
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
                        log_images_to_tensorboard(self.writer, student_pred, target, epoch, prefix="val_student")
                        log_images_to_tensorboard(self.writer, teacher_pred, target, epoch, prefix="val_teacher")
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
                import wandb
                log_dict = {"val/loss": avg_loss, "epoch": epoch}
                for key, val in avg_metrics.items():
                    log_dict[f"val/{key}"] = val
                wandb.log(log_dict, step=epoch)

        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None, is_best: bool = False):
        """Save student model checkpoint."""
        if not self.is_main_process:
            return

        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "distillation_temperature": self.temperature,
            "distillation_alpha": self.alpha
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save regular checkpoint
        if not self.config.training.save_best_only:
            checkpoint_path = checkpoint_dir / f"student_epoch_{epoch:04d}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved student checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_student.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best student model: {best_path}")

        # Save latest
        latest_path = checkpoint_dir / "latest_student.pth"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load student checkpoint to resume training."""
        print(f"Loading student checkpoint: {checkpoint_path}")
        try:
            with torch.serialization.safe_globals([TrainConfig]):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except pickle.UnpicklingError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = min(checkpoint["epoch"] + 1, self.config.training.epochs)
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        print(f"Resumed from epoch {self.current_epoch}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Student Training with Knowledge Distillation")
        print("="*80)

        for epoch in range(self.current_epoch, self.config.training.epochs):
            # Apply warmup scheduler
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                lr = self.warmup_start_lr + (self.config.optimizer.g_lr - self.warmup_start_lr) * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                if self.is_main_process:
                    print(f"Warmup: Epoch {epoch}, LR: {lr:.6f}")

            # Train
            avg_loss = self.train_epoch(train_loader, epoch)

            if self.is_main_process:
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Loss: {avg_loss:.4f}")

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

            # Step scheduler (skip during warmup)
            if epoch >= self.warmup_epochs:
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_loss is not None:
                            self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

        print("\n" + "="*80)
        print("Student Training Complete!")
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
    # Load student config
    if args.config == "default":
        config = get_default_config()
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
    if args.device:
        config.training.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir

    # Validate paths
    if not args.shards_dir:
        if not config.data.input_dir:
            raise ValueError("Input directory is not set. Pass --input-dir")
        if not config.data.output_dir:
            raise ValueError("Output directory is not set. Pass --output-dir")
        if not config.data.metadata_path:
            raise ValueError("Metadata path is not set. Pass --metadata-path")

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
        image_size=config.data.image_size,
        seed=config.training.seed,
        shards_dir=args.shards_dir
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
        image_size=config.data.image_size,
        seed=config.training.seed,
        shards_dir=args.shards_dir
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create trainer
    trainer = Trainer(
        config=config,
        teacher_checkpoint_path=args.teacher_checkpoint,
        temperature=args.temperature,
        alpha=args.alpha,
        rank=0
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student Model with Knowledge Distillation")

    # Required (unless shards are used)
    parser.add_argument("--teacher-checkpoint", type=str, default=None,
                      help="Path to pre-trained teacher model checkpoint (required unless --shards-dir is used)")

    # Config
    parser.add_argument("--config", type=str, default="default",
                      help="Config to use: 'default' or path to custom config (e.g., configs/mobilenetv3_2048.py)")

    # Distillation hyperparameters
    parser.add_argument("--temperature", type=float, default=4.0,
                      help="Temperature for distillation (default: 4.0)")
    parser.add_argument("--alpha", type=float, default=0.3,
                      help="Weight for distillation loss. Final loss = alpha*soft + (1-alpha)*hard (default: 0.3)")

    # Data
    parser.add_argument("--shards-dir", type=str, default=None,
                      help="Directory containing pre-computed distillation shards (.pt files)")
    parser.add_argument("--batch-size", type=int, default=None,
                      help="Batch size per GPU")
    parser.add_argument("--input-dir", type=str, default=None,
                      help="Directory containing rendered samples")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory containing ground-truth materials")
    parser.add_argument("--metadata-path", type=str, default=None,
                      help="Path to render_metadata.json")
    parser.add_argument("--render-curriculum", type=int, choices=[0, 1, 2], default=None,
                      help="0=clean only, 1=match dataset clean/dirty ratio, 2=dirty only")
    parser.add_argument("--device", type=str, default=None,
                      help="Device: 'auto' (default), 'cuda', 'cuda:0', or 'cpu'")

    # Training
    parser.add_argument("--epochs", type=int, default=None,
                      help="Number of epochs")
    parser.add_argument("--resume", type=str, default=None,
                      help="Path to student checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                      help="Directory to save student checkpoints")

    args = parser.parse_args()

    # Validation
    if not args.shards_dir and not args.teacher_checkpoint:
        parser.error("--teacher-checkpoint is required unless --shards-dir is specified.")

    main(args)
