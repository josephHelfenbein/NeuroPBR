"""
Configuration for Ultra-stable backbone at 512×512 resolution.

This config uses very conservative settings to prevent any collapse.
This is the default teacher config.

Usage:
    python train.py --config configs/ultra_stable.py \
        --input-dir /path/to/data/input \
        --output-dir /path/to/data/output
"""

from train_config import TrainConfig


def get_config():
    config = TrainConfig()

    # Model
    config.model.encoder_type = "resnet"
    config.model.encoder_backbone = "resnet50"
    # stride=2 gives a 16x16 latent per view at 512 input (768 cross-view ViT
    # tokens total). 16x less attention compute than stride=1 (32x32/view,
    # 3072 tokens) and easier on activation memory for the high-res early
    # layers. SR head (configured below) upsamples the decoder output back to 512.
    config.model.encoder_stride = 2
    config.model.freeze_backbone = False
    # Freeze pretrained ResNet BN: with a small per-GPU batch the BN running
    # stats are too noisy to be useful. freeze_bn=True keeps the well-calibrated
    # ImageNet statistics (encoder also overrides .train() to keep BN in eval).
    config.model.freeze_bn = True

    # Transformer for cross-view fusion
    config.model.use_transformer = True
    config.model.transformer_dim = 2048
    config.model.transformer_num_heads = 32
    config.model.transformer_depth = 4
    config.model.transformer_mlp_ratio = 4

    # Decoder
    config.model.decoder_type = "shared_heads"
    # decoder_sr_scale=2: with stride=2, decoder produces 256x256; SR head
    # upsamples to 512x512. Set explicitly here because TrainConfig.__post_init__
    # runs before this override is applied.
    config.model.decoder_sr_scale = 2

    # At 512 the ViT/decoder activations are small, so skip their recompute
    config.model.checkpoint_transformer = False
    config.model.checkpoint_decoder = False

    # GAN for realistic outputs (weakened to prevent mode collapse)
    config.model.use_gan = True
    config.model.discriminator_type = "configurable"
    config.model.discriminator_n_layers = 4
    config.model.discriminator_ndf = 64

    # Loss
    config.loss.w_l1 = 1.0
    config.loss.w_ssim = 0.3
    config.loss.w_normal = 2.0

    # Per-map L1 weights
    config.loss.w_albedo = 1.5
    config.loss.w_roughness = 1.0
    config.loss.w_metallic = 1.5
    config.loss.w_normal_map = 3.0
    
    config.loss.metallic_boost = 5.0
    config.loss.w_variance_match = 5.0
    config.loss.w_normal_xy = 10.0
    config.loss.w_color_mean = 5.0

    # Data
    config.data.image_size = (512, 512)
    config.data.output_size = (512, 512)
    config.data.batch_size = 4
    config.data.num_workers = 8
    config.data.prefetch_factor = 2
    config.data.pin_memory = True
    config.data.horizontal_flip = True
    config.data.vertical_flip = False

    # Transform: normalize input renders with ImageNet stats so the pretrained
    # ResNet encoder receives its expected input distribution. Target PBR maps
    # stay at 0.5/0.5 normalization (decoupled in train_config).
    config.transform.use_imagenet_stats = True
    # TrainConfig.__post_init__ already ran at TrainConfig() construction above,
    # so flipping use_imagenet_stats here does NOT retroactively update
    # input_mean/input_std. Set them explicitly so the encoder actually receives
    # ImageNet-normalized input.
    config.transform.input_mean = [0.485, 0.456, 0.406]
    config.transform.input_std = [0.229, 0.224, 0.225]

    # Optimizer
    config.optimizer.g_optimizer = "adamw"
    config.optimizer.g_lr = 1e-4
    config.optimizer.g_betas = (0.9, 0.999)
    config.optimizer.g_weight_decay = 1e-4
    
    # Discriminator optimizer (lower LR to reduce dominance over generator)
    config.optimizer.d_optimizer = "adamw"
    config.optimizer.d_lr = 2e-5
    config.optimizer.d_betas = (0.0, 0.9)
    config.optimizer.d_weight_decay = 1e-4

    # Scheduler with longer warmup
    config.optimizer.scheduler = "cosine"
    config.optimizer.scheduler_warmup_epochs = 10  # LONGER warmup
    config.optimizer.scheduler_min_lr = 1e-6

    # Training
    config.training.epochs = 100
    config.training.use_amp = True
    config.training.grad_clip_norm = 0.5

    # GAN schedule - start at epoch 35
    config.training.gan_start_epoch = 35
    config.training.d_steps_per_g_step = 1
    
    # GAN loss weight (reduced to prevent discriminator dominating reconstruction)
    config.loss.w_gan = 0.05

    # Checkpointing: latest.pth is always overwritten every epoch (resume
    # safety), best_model.pth is written whenever val loss improves, and
    # checkpoint_epoch_XXXX.pth snapshots are kept every 5 epochs (each
    # ~3.3 GB; every-epoch would be ~330 GB over a 100-epoch run).
    config.training.save_every_n_epochs = 5
    config.training.save_best_only = False

    # Logging
    config.training.log_every_n_steps = 10
    config.training.log_images_every_n_epochs = 1
    config.training.use_tensorboard = True

    # Reproducibility
    config.training.seed = 42

    return config
