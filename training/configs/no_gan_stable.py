"""
No-GAN stable config - reliable training without adversarial instability.

This config mirrors the default teacher setup (ResNet50 + ViT) but disables
the GAN entirely. Trade-offs:

PROS:
+ No mode collapse risk from discriminator failure
+ Monotonic convergence (no oscillation between G and D)
+ Faster per-epoch (~10-15% since no D forward/backward)
+ Suitable for distillation - student doesn't need GAN sharpness

CONS:
- Slightly "softer" textures (less high-frequency detail)
- May produce over-smoothed albedo in highly textured materials
- Metallic edges can be less crisp

EXPECTED QUALITY:
- PSNR: 22-26 dB (vs 24-28 with GAN)
- SSIM: 0.75-0.85 (similar to GAN)
- Normal angle: 8-12° (similar to GAN)

The L1 + SSIM + Normal loss combination provides strong reconstruction.
For most PBR workflows, the quality difference is negligible, especially
after the student distillation step.

RESOLUTION: 2048×2048 native (downsample to 1024 during shard generation)

RECOMMENDED USE:
1. Train teacher with this config (~100 epochs)
2. Generate 1024×1024 distillation shards with --shard-output-size 1024
3. Train student (which also doesn't use GAN)
4. If you need sharper textures, fine-tune with GAN for 10-20 epochs
   using a separate config with gan_start_epoch=0.
"""

from train_config import TrainConfig


def get_config():
    config = TrainConfig()

    # ========== Model (same as default) ==========
    config.model.encoder_type = "resnet"
    config.model.encoder_backbone = "resnet50"
    config.model.encoder_stride = 1  # Keep full resolution
    config.model.freeze_backbone = False
    config.model.freeze_bn = False

    # Transformer for cross-view fusion
    config.model.use_transformer = True
    config.model.transformer_dim = 2048
    config.model.transformer_num_heads = 32
    config.model.transformer_depth = 4
    config.model.transformer_mlp_ratio = 4

    # Decoder
    config.model.decoder_type = "shared_heads"
    config.model.decoder_sr_scale = 0  # No super-resolution

    # ========== GAN: DISABLED ==========
    config.model.use_gan = False  # Key difference!

    # ========== Loss (rebalanced for no-GAN) ==========
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.5       # Bumped from 0.3 - helps preserve structure
    config.loss.w_normal = 1.5    # Strong emphasis on normal angular loss
    config.loss.w_gan = 0.0       # No GAN

    # Per-map L1 weights
    # Metallic and normal need heavy boost because:
    # - Metallic: ~80% of materials are non-metallic, so constant 0 minimizes L1
    # - Normal: flat [0,0,1] is a "safe" output for smooth surfaces
    # Without boosting, these heads don't learn
    config.loss.w_albedo = 1.0
    config.loss.w_roughness = 1.0
    config.loss.w_metallic = 3.0     # Heavy boost - minority class
    config.loss.w_normal_map = 2.5   # Heavy boost - critical for shading
    
    # Sample-aware metallic boost: when a sample HAS metallic content (GT > 0.1),
    # multiply that sample's metallic loss by this factor. This compensates for
    # the ~80% non-metallic samples where pred=target=0 gives zero gradient.
    config.loss.metallic_boost = 10.0

    # ========== Data (2048×2048 native resolution) ==========
    config.data.image_size = (2048, 2048)   # Input resolution
    config.data.output_size = (2048, 2048)  # Output resolution
    config.data.batch_size = 1              # Safe for 2048² on most GPUs
    config.data.num_workers = 8
    config.data.render_curriculum = 0  # Clean only for stable training
    config.data.val_ratio = 0.1

    # Augmentation
    config.data.use_augmentation = True
    config.data.horizontal_flip = True
    config.data.vertical_flip = False

    # ========== Optimizer ==========
    config.optimizer.g_optimizer = "adamw"
    config.optimizer.g_lr = 2e-4
    config.optimizer.g_betas = (0.9, 0.999)  # Standard betas (not GAN-tuned)
    config.optimizer.g_weight_decay = 1e-4

    # No discriminator optimizer needed
    config.optimizer.d_lr = 0.0

    # Scheduler
    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 5
    config.optimizer.scheduler_min_lr = 1e-6

    # ========== Training ==========
    config.training.epochs = 100
    config.training.use_amp = True
    config.training.grad_clip_norm = 1.0

    # GAN settings (unused but set for safety)
    config.training.gan_start_epoch = 999  # Never
    config.training.d_steps_per_g_step = 0

    # Checkpointing - save every epoch for early validation
    config.training.save_every_n_epochs = 1
    config.training.save_best_only = False

    # Logging
    config.training.log_every_n_steps = 10
    config.training.log_images_every_n_epochs = 5
    config.training.use_tensorboard = True

    # Reproducibility
    config.training.seed = 42

    return config
