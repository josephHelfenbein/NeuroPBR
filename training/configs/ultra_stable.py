"""
Ultra-stable config - prioritizes stable training over speed.

This config uses very conservative settings to prevent any collapse:
- Lower learning rate (1e-4 instead of 2e-4)
- Longer warmup (10 epochs)  
- Balanced loss weights (no aggressive boosting)
- Moderate metallic boost (5x instead of 10x)

The previous config had issues with:
1. Roughness and normal heads collapsing while metallic improved
2. BatchNorm instabilities (extreme variances)
3. The variance regularization was fighting with L1 loss

STRATEGY:
- Equal treatment of all heads initially
- Let the model learn basic reconstructions first
- Metallic boost is moderate (5x) to avoid destabilizing other heads
"""

from train_config import TrainConfig


def get_config():
    config = TrainConfig()

    # ========== Model ==========
    config.model.encoder_type = "resnet"
    config.model.encoder_backbone = "resnet50"
    config.model.encoder_stride = 1
    config.model.freeze_backbone = False
    config.model.freeze_bn = False  # Keep BN trainable

    # Transformer for cross-view fusion
    config.model.use_transformer = True
    config.model.transformer_dim = 2048
    config.model.transformer_num_heads = 32
    config.model.transformer_depth = 4
    config.model.transformer_mlp_ratio = 4

    # Decoder
    config.model.decoder_type = "shared_heads"
    config.model.decoder_sr_scale = 0

    # No GAN
    config.model.use_gan = False

    # ========== Loss (conservative, balanced) ==========
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.3       # Lower SSIM weight for stability
    config.loss.w_normal = 2.0     # INCREASED: Normal angular loss (was 1.0)
    config.loss.w_gan = 0.0

    # Per-map L1 weights - boost normals significantly
    config.loss.w_albedo = 1.0
    config.loss.w_roughness = 1.0
    config.loss.w_metallic = 1.5   # Slight boost only
    config.loss.w_normal_map = 3.0 # INCREASED: Normal L1 loss (was 1.5)
    
    # Moderate metallic sample boost (5x instead of 10x)
    config.loss.metallic_boost = 5.0
    
    # Variance matching: penalize when pred variance < target variance
    # This prevents mode collapse where model outputs constant values
    # Applied to roughness and normal maps
    config.loss.w_variance_match = 5.0  # Strong variance enforcement

    # ========== Data ==========
    config.data.image_size = (2048, 2048)
    config.data.output_size = (2048, 2048)
    config.data.num_views = 3
    config.data.batch_size = 2
    config.data.num_workers = 8
    config.data.prefetch_factor = 2
    config.data.pin_memory = True
    config.data.horizontal_flip = True
    config.data.vertical_flip = False

    # ========== Optimizer (CONSERVATIVE) ==========
    config.optimizer.g_optimizer = "adamw"
    config.optimizer.g_lr = 1e-4        # LOWER LR for stability
    config.optimizer.g_betas = (0.9, 0.999)
    config.optimizer.g_weight_decay = 1e-4

    # Scheduler with longer warmup
    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 10  # LONGER warmup
    config.optimizer.scheduler_min_lr = 1e-6

    # ========== Training ==========
    config.training.epochs = 100
    config.training.use_amp = True
    config.training.grad_clip_norm = 0.5  # TIGHTER gradient clipping

    # GAN disabled
    config.training.gan_start_epoch = 999
    config.training.d_steps_per_g_step = 0

    # Save every epoch for monitoring
    config.training.save_every_n_epochs = 1
    config.training.save_best_only = False

    # Logging
    config.training.log_every_n_steps = 10
    config.training.log_images_every_n_epochs = 1  # Log images every epoch for debugging
    config.training.use_tensorboard = True

    # Reproducibility
    config.training.seed = 42

    return config
