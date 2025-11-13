"""
Configuration for MobileNetV3-Large backbone at 2048×2048 resolution.

Optimized for mobile deployment on 4GB iPhone.
Based on memory profiling showing ~100 MB usage on Mac (FP32).

Usage:
    python train.py --config configs/mobilenetv3_2048.py --data-root /path/to/data
"""

from train_config import TrainConfig


def get_config() -> TrainConfig:
    """Configuration for MobileNetV3-Large training at 2048×2048."""
    config = TrainConfig()

    # Data: 2048×2048 images
    config.data.image_size = (2048, 2048)
    config.data.output_size = (2048, 2048)
    config.data.batch_size = 2  # Reduced for high resolution
    config.data.num_workers = 8

    # Model: MobileNetV3-Large encoder
    config.model.encoder_type = "mobilenetv3"
    config.model.encoder_backbone = "mobilenet_v3_large"
    config.model.encoder_stride = 1  # 2048 → 2048 (no downsampling)
    config.model.freeze_backbone = False  # Fine-tune the backbone
    config.model.freeze_bn = False

    # Transformer: Use 16 heads for compatibility (can use 8, 16, 24, 28, 32)
    config.model.transformer_num_heads = 16
    config.model.transformer_depth = 4

    # Discriminator: 6-layer for high resolution
    config.model.discriminator_type = "configurable"
    config.model.discriminator_n_layers = 6
    config.model.discriminator_use_sigmoid = False  # For hinge loss

    # Loss: Balanced reconstruction + GAN
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.3
    config.loss.w_normal = 0.5
    config.loss.w_gan = 0.05
    config.loss.gan_loss_type = "hinge"

    # Training
    config.training.epochs = 100
    config.training.gan_start_epoch = 5
    config.training.val_every_n_epochs = 1
    config.training.save_every_n_epochs = 5
    config.training.use_amp = True  # Mixed precision for faster training

    # Optimizer
    config.optimizer.g_lr = 2e-4
    config.optimizer.d_lr = 2e-4
    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 5

    return config


if __name__ == "__main__":
    # Print config for verification
    config = get_config()
    print("MobileNetV3-Large 2048×2048 Config:")
    print(f"  Image size: {config.data.image_size}")
    print(f"  Batch size: {config.data.batch_size}")
    print(
        f"  Encoder: {config.model.encoder_type} ({config.model.encoder_backbone})")
    print(f"  Encoder stride: {config.model.encoder_stride}")
    print(f"  Transformer heads: {config.model.transformer_num_heads}")
    print(f"  Discriminator layers: {config.model.discriminator_n_layers}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning rate: {config.optimizer.g_lr}")
