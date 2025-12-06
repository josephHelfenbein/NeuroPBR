"""
Configuration for ConvAttn Student Model.

Implements the ESC (Efficient Scalable Convolution) paper's PLK approach:
- MobileNetV3-Large encoder (pretrained from ImageNet)
- PLK (Pre-computed Large Kernel) bottleneck replacing transformer
- Shared 17×17 learnable kernel across all blocks
- SE-style channel attention for instance-adaptive weighting

Key differences from transformer-based student:
- Replaces O(N²) self-attention with O(N) PLK convolution
- Much lower memory allows higher resolution (768+ vs 512 for ViT)
- Bottleneck C=320 for balance between quality and ANE compatibility
- Feature distillation at bottleneck + two decoder skips (λ_feat=0.1)

Training strategy:
- Output L1 + Feature L2 distillation
- Initialize encoder from MobileNetV3 pretrained weights
- Mixed-precision training (AMP)
- Cosine LR schedule with warmup

Usage:
    python student/train.py --config configs/convattn_student.py \\
        --shards-dir teacher_shards_1024 \\
        --input-dir /path/to/data/input \\
        --output-dir /path/to/data/output

With feature distillation (requires teacher checkpoint):
    python student/train.py --config configs/convattn_student.py \\
        --teacher-checkpoint /path/to/teacher.pth \\
        --input-dir /path/to/data/input \\
        --output-dir /path/to/data/output \\
        --use-feature-distillation
"""

from train_config import TrainConfig


# ConvAttn-specific configuration (PLK-based)
CONVATTN_CONFIG = {
    # Bottleneck channels (matches decoder expected, ANE-friendly)
    "bottleneck_channels": 320,
    
    # Number of PLK blocks (2-4, 3 is sweet spot)
    "num_convattn_blocks": 3,
    
    # PLK kernel size (17 gives ~17×17 receptive field per block)
    "kernel_size": 17,
    
    # Feature distillation weight
    "lambda_feat": 0.1,
    
    # Output distillation weight  
    "lambda_output": 1.0,
    
    # Distillation temperature
    "temperature": 4.0,
    
    # Alpha for soft vs hard targets
    "alpha": 0.3,
}


def get_config() -> TrainConfig:
    """Configuration for ConvAttn student training at 1024×1024 (no SR)."""
    config = TrainConfig()

    # ========== Data Configuration ==========
    # 1024×1024 input and output (no SR head)
    # MobileNetV3 with stride=2 produces 32×32 latent from 1024 input
    # 5 decoder upsamples (32→1024), no SR needed
    config.data.image_size = (1024, 1024)
    config.data.output_size = (1024, 1024)
    config.data.batch_size = 4  # Lower batch size for higher resolution
    config.data.num_workers = 8
    config.data.pin_memory = True
    config.data.persistent_workers = True

    # ========== Model Configuration ==========
    # MobileNetV3-Large encoder
    config.model.encoder_type = "mobilenetv3"
    config.model.encoder_backbone = "mobilenet_v3_large"
    
    # Encoder stride=2: 1024 → 512 → 256 → 128 → 64 → 32 latent
    config.model.encoder_stride = 2
    
    # No SR head: Decoder upsamples directly to 1024×1024
    # (32→64→128→256→512→1024 via decoder, no SR)
    config.model.decoder_sr_scale = 1
    
    # Don't freeze backbone - allow fine-tuning
    config.model.freeze_backbone = False
    config.model.freeze_bn = False
    
    # Transformer settings (not used for ConvAttn, but kept for compatibility)
    # The ConvAttn module replaces the transformer
    config.model.use_transformer = False  # ConvAttn replaces this
    config.model.transformer_dim = 320  # Used as reference for bottleneck
    config.model.transformer_num_heads = 4
    config.model.transformer_depth = 3
    
    # No discriminator for student training
    config.model.use_gan = False
    config.model.discriminator_type = "configurable"
    config.model.discriminator_n_layers = 4
    config.model.discriminator_use_sigmoid = False

    # ========== Loss Configuration ==========
    # Reconstruction losses (used in hard target loss)
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.3
    config.loss.w_normal = 0.5
    config.loss.w_gan = 0.0  # No GAN for student
    
    # Per-map weights
    config.loss.w_albedo = 1.0
    config.loss.w_roughness = 1.0
    config.loss.w_metallic = 1.0
    config.loss.w_normal_map = 1.0
    
    config.loss.gan_loss_type = "hinge"

    # ========== Training Configuration ==========
    config.training.epochs = 100
    config.training.gan_start_epoch = 999  # Never start GAN
    config.training.val_every_n_epochs = 1
    config.training.save_every_n_epochs = 5
    config.training.log_every_n_steps = 50
    config.training.log_images_every_n_epochs = 5
    
    # Mixed precision for memory efficiency
    config.training.use_amp = True
    
    # Gradient clipping for stability
    config.training.grad_clip_norm = 1.0
    
    # Logging
    config.training.use_tensorboard = True
    config.training.use_wandb = False
    config.training.wandb_project = "neuropbr-student"

    # ========== Optimizer Configuration ==========
    # AdamW with lower learning rate for fine-tuning
    config.optimizer.g_optimizer = "adamw"
    config.optimizer.g_lr = 1e-4  # Lower than teacher for fine-tuning
    config.optimizer.g_betas = (0.9, 0.999)
    config.optimizer.g_weight_decay = 1e-4
    
    # Cosine schedule with warmup
    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 5
    config.optimizer.scheduler_min_lr = 1e-6

    return config


def get_convattn_config() -> dict:
    """Get ConvAttn-specific configuration parameters."""
    return CONVATTN_CONFIG.copy()


def get_high_quality_config() -> TrainConfig:
    """
    Higher quality configuration with more blocks and longer training.
    
    Use when compute budget allows and maximum quality is needed.
    """
    config = get_config()
    
    # More training
    config.training.epochs = 150
    
    # Lower learning rate for more careful training
    config.optimizer.g_lr = 5e-5
    config.optimizer.scheduler_warmup_epochs = 10
    
    return config


def get_fast_iteration_config() -> TrainConfig:
    """
    Fast iteration configuration for testing and debugging.
    
    Smaller model, fewer epochs for quick experiments.
    """
    config = get_config()
    
    # Smaller batch for faster iterations
    config.data.batch_size = 4
    
    # Fewer epochs
    config.training.epochs = 20
    config.training.val_every_n_epochs = 2
    config.training.save_every_n_epochs = 5
    
    # Less warmup
    config.optimizer.scheduler_warmup_epochs = 2
    
    return config


def get_convattn_params():
    """
    Get the ConvAttn-specific parameters for model creation.
    
    Returns dict with:
        - bottleneck_channels: 320
        - num_blocks: 3
        - kernel_size: 17
        - lambda_feat: 0.1
        - lambda_output: 1.0
        - temperature: 4.0
        - alpha: 0.3
    """
    return {
        "bottleneck_channels": CONVATTN_CONFIG["bottleneck_channels"],
        "num_blocks": CONVATTN_CONFIG["num_convattn_blocks"],
        "kernel_size": CONVATTN_CONFIG["kernel_size"],
        "lambda_feat": CONVATTN_CONFIG["lambda_feat"],
        "lambda_output": CONVATTN_CONFIG["lambda_output"],
        "temperature": CONVATTN_CONFIG["temperature"],
        "alpha": CONVATTN_CONFIG["alpha"],
    }


if __name__ == "__main__":
    # Print config for verification
    config = get_config()
    convattn_params = get_convattn_params()
    
    print("ConvAttn Student Configuration (PLK-based):")
    print("=" * 50)
    print()
    print("Data:")
    print(f"  Input size: {config.data.image_size}")
    print(f"  Output size: {config.data.output_size} (after SR 4×)")
    print(f"  Batch size: {config.data.batch_size}")
    print()
    print("Model:")
    print(f"  Encoder: {config.model.encoder_type} ({config.model.encoder_backbone})")
    print(f"  Encoder stride: {config.model.encoder_stride}")
    print(f"  SR scale: {config.model.decoder_sr_scale}×")
    print(f"  Use transformer: {config.model.use_transformer} (PLK ConvAttn replaces it)")
    print()
    print("PLK Bottleneck:")
    print(f"  Bottleneck channels: {convattn_params['bottleneck_channels']}")
    print(f"  Number of PLK blocks: {convattn_params['num_blocks']}")
    print(f"  PLK kernel size: {convattn_params['kernel_size']}×{convattn_params['kernel_size']}")
    print()
    print("Distillation:")
    print(f"  λ_output (L1): {convattn_params['lambda_output']}")
    print(f"  λ_feat (L2): {convattn_params['lambda_feat']}")
    print(f"  Temperature: {convattn_params['temperature']}")
    print(f"  Alpha (soft vs hard): {convattn_params['alpha']}")
    print()
    print("Training:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning rate: {config.optimizer.g_lr}")
    print(f"  Scheduler: {config.optimizer.scheduler}")
    print(f"  Mixed precision: {config.training.use_amp}")
    print()
    print("Inference pipeline:")
    print("  512×512 input → PLK ConvAttn Student → 1024×1024 → Lanczos → 2048×2048")
