"""
High quality config - best results but slower training.

Features:
- ResNet101 encoder (large capacity)
- 6-layer transformer (deep fusion)
- Higher loss weights for quality
"""

from train_config import TrainConfig

def get_config():
    config = TrainConfig()
    
    # Model
    config.model.encoder_backbone = "resnet101"
    config.model.encoder_stride = 2
    config.model.transformer_depth = 6
    config.model.transformer_num_heads = 32
    config.model.use_gan = True
    
    # Loss
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.5
    config.loss.w_normal = 0.8
    config.loss.w_gan = 0.1
    
    # Emphasize albedo quality
    config.loss.w_albedo = 1.2
    
    # Training
    config.training.epochs = 200
    config.training.gan_start_epoch = 10  # More warmup
    config.training.use_amp = True
    config.training.grad_clip_norm = 1.0
    
    # Data
    config.data.image_size = (2048, 2048)
    config.data.batch_size = 1  # Reduced for 2048x2048
    config.data.num_workers = 8
    
    # Optimizer
    config.optimizer.g_lr = 1e-4
    config.optimizer.d_lr = 2e-4
    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 10
    
    return config
