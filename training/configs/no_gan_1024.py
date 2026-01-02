"""
No-GAN stable config at 1024×1024 resolution.

Same as no_gan_stable.py but at 1024×1024 for faster training and
native shard resolution.

Training time: ~3-4 days (vs ~9 days at 2048²)
"""

from train_config import TrainConfig


def get_config():
    config = TrainConfig()

    # ========== Model ==========
    config.model.encoder_type = "resnet"
    config.model.encoder_backbone = "resnet50"
    config.model.encoder_stride = 1
    config.model.freeze_backbone = False
    config.model.freeze_bn = False

    # Transformer for cross-view fusion
    config.model.use_transformer = True
    config.model.transformer_dim = 2048
    config.model.transformer_num_heads = 32
    config.model.transformer_depth = 4
    config.model.transformer_mlp_ratio = 4

    # Decoder - NO super-resolution (1024→1024)
    config.model.decoder_type = "shared_heads"
    config.model.decoder_sr_scale = 0

    # GAN disabled
    config.model.use_gan = False

    # ========== Loss ==========
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.5
    config.loss.w_normal = 0.7
    config.loss.w_gan = 0.0

    config.loss.w_albedo = 1.0
    config.loss.w_roughness = 1.0
    config.loss.w_metallic = 1.0
    config.loss.w_normal_map = 1.2

    # ========== Data: 1024×1024 ==========
    config.data.image_size = (1024, 1024)
    config.data.output_size = (1024, 1024)
    config.data.batch_size = 2  # Conservative - let auto-scaler increase
    config.data.num_workers = 8
    config.data.render_curriculum = 0
    config.data.val_ratio = 0.1

    config.data.use_augmentation = True
    config.data.horizontal_flip = True
    config.data.vertical_flip = False

    # ========== Optimizer ==========
    config.optimizer.g_optimizer = "adamw"
    # Base LR for batch_size=2. If auto-scaler increases batch, LR should scale too.
    # For batch=10 (5× larger), effective LR would be 5e-4
    # The training script should handle this if it has LR scaling logic.
    # If not, we set a middle-ground LR that works for larger batches:
    config.optimizer.g_lr = 5e-4  # Works for batch 8-12
    config.optimizer.g_betas = (0.9, 0.999)
    config.optimizer.g_weight_decay = 1e-4

    config.optimizer.d_lr = 0.0

    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 10  # Longer warmup helps with higher LR
    config.optimizer.scheduler_min_lr = 1e-6

    # ========== Training ==========
    config.training.epochs = 100
    config.training.use_amp = True
    config.training.grad_clip_norm = 0.5  # Tighter clipping

    config.training.gan_start_epoch = 999
    config.training.d_steps_per_g_step = 0

    config.training.save_every_n_epochs = 5
    config.training.save_best_only = False

    config.training.log_every_n_steps = 10
    config.training.log_images_every_n_epochs = 5
    config.training.use_tensorboard = True

    config.training.seed = 42

    return config
