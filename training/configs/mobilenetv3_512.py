"""
Configuration for MobileNetV3-Large backbone at 512×512 resolution.

Optimized for iPhone deployment with ANE constraints.
Train/inference resolution match eliminates domain gap.

Pipeline:
    Training:  512×512 input → Model → 1024×1024 output (SR 2×)
    Inference: 512×512 input → Model → 1024×1024 → Lanczos → 2048×2048

Shard Generation:
    Generate shards at 1024×1024 to match student output resolution:
    
    python teacher_infer.py \\
        --checkpoint /path/to/teacher.pth \\
        --data-root /path/to/data \\
        --out-dir teacher_shards_1024 \\
        --shard-output-size 1024

Usage:
    python student/train.py --config configs/mobilenetv3_512.py \\
        --shards-dir teacher_shards_1024 \\
        --input-dir /path/to/data/input \\
        --output-dir /path/to/data/output
"""

from train_config import TrainConfig


def get_config() -> TrainConfig:
    """Configuration for MobileNetV3-Large training at 512×512."""
    config = TrainConfig()

    # Data: 512×512 input, 1024×1024 output (after SR)
    # Dataset will load 2048 images and resize:
    #   - Inputs: 2048 → 512 (4× downsample)
    #   - Targets: 2048 → 1024 (2× downsample, matches SR output)
    config.data.image_size = (512, 512)
    config.data.output_size = (1024, 1024)  # SR head doubles resolution
    config.data.batch_size = 8  # Can use larger batch at lower resolution
    config.data.num_workers = 8

    # Model: MobileNetV3-Large encoder
    config.model.encoder_type = "mobilenetv3"
    config.model.encoder_backbone = "mobilenet_v3_large"
    
    # Encoder stride=2: 512 input → 256 → 128 → 64 → 32 → 16 latent
    # This keeps latent small for transformer efficiency
    config.model.encoder_stride = 2
    
    # SR 2×: Decoder outputs 512, SR head upscales to 1024
    # This gives us better quality than raw 512 output
    config.model.decoder_sr_scale = 2

    config.model.freeze_backbone = False
    config.model.freeze_bn = False

    # Transformer: Same lightweight config as 2048 version
    # Works well for mobile deployment
    config.model.use_transformer = True
    config.model.transformer_dim = 256
    config.model.transformer_num_heads = 4
    config.model.transformer_depth = 2
    config.model.transformer_mlp_ratio = 2

    # Discriminator: 4 layers sufficient for 1024 output
    # (fewer than 2048 version since output is smaller)
    config.model.discriminator_type = "configurable"
    config.model.discriminator_n_layers = 4
    config.model.discriminator_use_sigmoid = False  # For hinge loss

    # Loss: Same balanced weights
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.3
    config.loss.w_normal = 0.5
    config.loss.w_gan = 0.05
    config.loss.gan_loss_type = "hinge"

    # Training: Can train longer at lower resolution
    config.training.epochs = 100
    config.training.gan_start_epoch = 5
    config.training.val_every_n_epochs = 1
    config.training.save_every_n_epochs = 5
    config.training.use_amp = True  # Mixed precision

    # Optimizer
    config.optimizer.g_lr = 2e-4
    config.optimizer.d_lr = 2e-4
    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 5

    return config


if __name__ == "__main__":
    # Print config for verification
    config = get_config()
    print("MobileNetV3-Large 512×512 Config (Mobile-Optimized):")
    print(f"  Input size: {config.data.image_size}")
    print(f"  Output size: {config.data.output_size} (after SR 2×)")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Encoder: {config.model.encoder_type} ({config.model.encoder_backbone})")
    print(f"  Encoder stride: {config.model.encoder_stride}")
    print(f"  SR scale: {config.model.decoder_sr_scale}×")
    print(f"  Transformer dim: {config.model.transformer_dim}")
    print(f"  Discriminator layers: {config.model.discriminator_n_layers}")
    print(f"  Epochs: {config.training.epochs}")
    print()
    print("Inference pipeline:")
    print("  512×512 input → Model → 1024×1024 → Lanczos → 2048×2048")
