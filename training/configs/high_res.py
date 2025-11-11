"""
Quick config example for 2048x2048 training with 6-layer discriminator.

Copy this to configs/high_res.py and use with:
    python train.py --config configs/high_res.py --data-root /path/to/data
"""

from train_config import TrainConfig, DataConfig, ModelConfig, LossConfig, OptimizerConfig, TrainingConfig


def get_config() -> TrainConfig:
    """Configuration for high-resolution 2048x2048 training with 6-layer discriminator."""
    config = TrainConfig()
    
    # Data: 2048x2048 images
    config.data.image_size = (2048, 2048)
    config.data.output_size = (2048, 2048)
    config.data.batch_size = 2  # Reduced for memory
    config.data.num_workers = 8
    
    # Model: 6-layer discriminator
    config.model.encoder_stride = 1  # 2048 → 2048
    config.model.discriminator_type = "configurable"
    config.model.discriminator_n_layers = 6  # Large receptive field
    config.model.discriminator_use_sigmoid = False  # For hinge loss
    config.model.discriminator_ndf = 64
    
    # Loss: Balanced
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
    config.training.log_images_every_n_epochs = 5
    
    # Optimizer
    config.optimizer.g_lr = 2e-4
    config.optimizer.d_lr = 2e-4
    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 5
    
    return config


if __name__ == "__main__":
    # Print config for verification
    config = get_config()
    print("High-Resolution 2048x2048 Config:")
    print(f"  Image size: {config.data.image_size}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Encoder stride: {config.model.encoder_stride}")
    print(f"  Discriminator: {config.model.discriminator_type}")
    print(f"  Discriminator layers: {config.model.discriminator_n_layers}")
    print(f"  Epochs: {config.training.epochs}")
