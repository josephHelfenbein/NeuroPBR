"""
Extended training configuration for NeuroPBR multi-view fusion GAN training.
"""

from dataclasses import dataclass, field
from typing import Literal, List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""
    input_dir: Optional[str] = "./data/input"
    output_dir: Optional[str] = "./data/output"
    metadata_path: Optional[str] = None
    # Input size to encoder (what dataset loads/resizes to)
    image_size: tuple = (1024, 1024)
    # Output size from decoder (achieved via SR scale in decoder)
    output_size: tuple = (1024, 1024)
    batch_size: int = 2  # Per GPU (increase if GPU memory allows at 1024²)
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Dataset options
    # If True, train on dirty renders; otherwise use clean renders (default)
    use_dirty_renders: bool = False
    val_ratio: float = 0.1  # Train/val split ratio

    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    random_rotation: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Architecture type
    architecture: Literal["multiview_fusion",
                          "single_view", "simple_unet"] = "multiview_fusion"

    # Encoder settings
    encoder_type: Literal["unet", "unet_stride",
                          "resnet", "mobilenetv3"] = "resnet"
    encoder_backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101",
                              "resnet152", "mobilenet_v3_large", "mobilenet_v3_small"] = "resnet50"
    encoder_in_channels: int = 3  # RGB input per view
    encoder_stride: Literal[1, 2] = 1  # 1 keeps 1024→1024, 2 halves to 512
    encoder_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 1024, 2048])
    freeze_backbone: bool = False
    freeze_bn: bool = False

    # Transformer settings (for multi-view fusion)
    use_transformer: bool = True
    transformer_dim: int = 2048
    transformer_num_heads: int = 32
    transformer_depth: int = 4
    transformer_mlp_ratio: int = 4
    transformer_proj_drop: float = 0.1
    transformer_attn_drop: float = 0.1
    transformer_drop_path_rate: float = 0.1

    # Decoder settings
    # Multi-head for 4 PBR maps
    decoder_type: Literal["shared_heads", "single"] = "shared_heads"
    decoder_skip_channels: List[int] = field(
        default_factory=lambda: [1024, 512, 256, 64])  # ResNet skips
    decoder_sr_scale: Literal[0, 2, 4] = 0  # No SR needed for 1024×1024 outputs

    # Output channels for each PBR map
    # albedo, roughness, metallic, normal
    output_channels: List[int] = field(default_factory=lambda: [3, 1, 1, 3])

    # GAN settings
    use_gan: bool = True
    # "simple" = losses.py, "configurable" = gan/discriminator.py
    discriminator_type: Literal["simple", "configurable"] = "configurable"
    # 3 (albedo) + 1 (roughness) + 1 (metallic) + 3 (normal)
    discriminator_in_channels: int = 8
    discriminator_ndf: int = 64
    # Number of layers in discriminator (more layers = larger receptive field)
    discriminator_n_layers: int = 6
    # False for hinge loss, True for BCE loss
    discriminator_use_sigmoid: bool = False


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: Literal["hybrid", "l1_only", "custom"] = "hybrid"

    # Loss weights for HybridLoss
    w_l1: float = 1.0
    w_ssim: float = 0.3
    w_normal: float = 0.5
    w_perceptual: float = 0.0  # Set > 0 to enable VGG perceptual loss
    w_gan: float = 0.05  # Generator GAN loss weight

    # Individual map weights for L1
    w_albedo: float = 1.0
    w_roughness: float = 1.0
    w_metallic: float = 1.0
    w_normal_map: float = 1.0

    # GAN loss type
    gan_loss_type: Literal["hinge", "bce"] = "hinge"

    # Discriminator loss weight
    w_discriminator: float = 1.0

    # Use perceptual loss
    use_perceptual: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler configuration."""
    # Generator optimizer
    g_optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    g_lr: float = 2e-4
    g_betas: tuple = (0.5, 0.999)
    g_weight_decay: float = 1e-4

    # Discriminator optimizer
    d_optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    d_lr: float = 2e-4
    d_betas: tuple = (0.5, 0.999)
    d_weight_decay: float = 0.0

    # Learning rate scheduler
    scheduler: Optional[Literal["cosine",
                                "step", "plateau", "none"]] = "cosine"
    scheduler_warmup_epochs: int = 5
    scheduler_min_lr: float = 1e-6

    # For StepLR
    step_size: int = 30
    gamma: float = 0.1

    # For ReduceLROnPlateau
    patience: int = 10
    factor: float = 0.5


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    epochs: int = 100
    start_epoch: int = 0

    # Device selection
    device: str = "auto"

    # Mixed precision training
    use_amp: bool = True

    # Gradient clipping
    grad_clip_norm: Optional[float] = 1.0

    # GAN training schedule
    gan_start_epoch: int = 5  # Start GAN training after N epochs
    d_steps_per_g_step: int = 1  # How many D updates per G update

    # Validation
    val_every_n_epochs: int = 1
    val_batches: Optional[int] = None  # None = validate on full val set

    # Checkpointing
    save_every_n_epochs: int = 5
    save_best_only: bool = False
    checkpoint_dir: str = "./checkpoints"

    # Logging
    log_every_n_steps: int = 10
    log_images_every_n_epochs: int = 5
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "neuropbr"
    wandb_run_name: Optional[str] = None

    # Resume training
    resume_from: Optional[str] = None  # Path to checkpoint

    # Multi-GPU
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    dist_backend: str = "nccl"
    dist_url: str = "env://"

    # Reproducibility
    seed: int = 42
    deterministic: bool = False


@dataclass
class TransformConfig:
    """Data normalization configuration."""
    # Normalization stats
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    # Use ImageNet pretrained stats for ResNet
    # If True: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    use_imagenet_stats: bool = False


@dataclass
class TrainConfig:
    """Main training configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)

    def __post_init__(self):
        """Validate and auto-adjust configuration."""
        if self.data.metadata_path is None and self.data.input_dir:
            self.data.metadata_path = str(Path(self.data.input_dir) / "render_metadata.json")

        # Ensure output_size defaults to image_size when unspecified
        if self.data.output_size is None:
            self.data.output_size = self.data.image_size

        # Adjust decoder SR scale so default configs stay at 1024×1024
        if self.model.encoder_stride == 1:
            self.model.decoder_sr_scale = 0  # Keep 1024 resolution
        elif self.model.encoder_stride == 2:
            self.model.decoder_sr_scale = 2  # Upsample 512 → 1024

        # Update skip channels based on encoder type
        if self.model.encoder_type == "resnet":
            self.model.decoder_skip_channels = [1024, 512, 256, 64]
        else:
            # UNet encoders
            channels = self.model.encoder_channels
            self.model.decoder_skip_channels = channels[::-1]  # Reverse

        # Set ImageNet normalization if using pretrained ResNet
        if self.transform.use_imagenet_stats:
            self.transform.mean = [0.485, 0.456, 0.406]
            self.transform.std = [0.229, 0.224, 0.225]

    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "optimizer": self.optimizer.__dict__,
            "training": self.training.__dict__,
            "transform": self.transform.__dict__,
        }


def get_default_config() -> TrainConfig:
    """Get default configuration for multi-view fusion GAN training."""
    return TrainConfig()


def get_quick_test_config() -> TrainConfig:
    """Get configuration for quick testing (small model, few epochs)."""
    config = TrainConfig()

    # Smaller model
    config.model.encoder_backbone = "resnet18"
    config.model.transformer_depth = 2
    config.model.transformer_num_heads = 16

    # Fewer epochs
    config.training.epochs = 10
    config.training.val_every_n_epochs = 2
    config.training.save_every_n_epochs = 5

    # Smaller batch
    config.data.batch_size = 2
    config.data.num_workers = 4

    return config


def get_lightweight_config() -> TrainConfig:
    """Get configuration for training without GAN (faster, simpler)."""
    config = TrainConfig()

    # Disable GAN
    config.model.use_gan = False
    config.loss.w_gan = 0.0

    # Focus on reconstruction losses
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.5
    config.loss.w_normal = 0.7

    return config


if __name__ == "__main__":
    # Test config creation
    config = get_default_config()
    print("Default Config:")
    print(f"  Architecture: {config.model.architecture}")
    print(
        f"  Encoder: {config.model.encoder_type} ({config.model.encoder_backbone})")
    print(f"  Transformer: {config.model.use_transformer}")
    print(f"  GAN: {config.model.use_gan}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Decoder SR scale: {config.model.decoder_sr_scale}")
