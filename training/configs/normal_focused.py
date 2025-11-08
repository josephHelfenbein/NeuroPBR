"""
Normal-focused config - optimized for high-quality normal maps.

Features:
- Heavy weighting on normal losses
- GAN training for sharper details
- ResNet50 balanced architecture
"""

from train_config import TrainConfig

def get_config():
    config = TrainConfig()
    
    # Model
    config.model.encoder_backbone = "resnet50"
    config.model.encoder_stride = 2
    config.model.transformer_depth = 4
    config.model.use_gan = True
    
    # Loss - Emphasize Normals
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.2  # Lower SSIM (only for albedo)
    config.loss.w_normal = 1.5  # HIGH angular consistency loss
    config.loss.w_gan = 0.08
    
    # Per-map weights - EMPHASIZE NORMAL
    config.loss.w_albedo = 0.8
    config.loss.w_roughness = 0.8
    config.loss.w_metallic = 0.8
    config.loss.w_normal_map = 2.0  # Double weight for normal L1
    
    # Training
    config.training.epochs = 150
    config.training.gan_start_epoch = 8
    config.training.use_amp = True
    
    # Data
    config.data.batch_size = 4
    
    # Optimizer
    config.optimizer.g_lr = 1.5e-4
    config.optimizer.d_lr = 2e-4
    config.optimizer.scheduler = "cosine"
    
    return config
