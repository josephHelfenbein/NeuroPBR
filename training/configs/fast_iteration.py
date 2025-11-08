"""
Fast iteration config - quick experiments and debugging.

Features:
- ResNet18 encoder (fast)
- 2-layer transformer (minimal)
- No perceptual loss
- Small batch size
"""

from train_config import TrainConfig

def get_config():
    config = TrainConfig()
    
    # Model
    config.model.encoder_backbone = "resnet18"
    config.model.encoder_stride = 2
    config.model.transformer_depth = 2
    config.model.transformer_num_heads = 16
    config.model.use_gan = True

    # Loss
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.3
    config.loss.w_normal = 0.5
    config.loss.w_perceptual = 0.0
    config.loss.w_gan = 0.05
    config.loss.use_perceptual = False

    # Training
    config.training.epochs = 50
    config.training.gan_start_epoch = 5
    config.training.use_amp = True
    config.training.val_every_n_epochs = 2
    config.training.save_every_n_epochs = 10

    # Data
    config.data.batch_size = 2
    config.data.num_workers = 4
    
    # Optimizer
    config.optimizer.g_lr = 2e-4
    config.optimizer.d_lr = 2e-4
    config.optimizer.scheduler = "cosine"
    
    return config
