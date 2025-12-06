"""
Teacher inference script for precomputed distillation.

This runs a trained teacher generator over the dataset once and saves its
outputs (PBR maps) into compact `.pt` shards for student distillation.

Example usage from the `training` directory:

    # Using default config, clean renders, 2048x2048 (from config)
    python teacher_infer.py \
        --data-root /path/to/data \
        --checkpoint checkpoints/best_model.pth

    # Explicit input/output dirs and dirty renders
    python teacher_infer.py \
        --input-dir /path/to/data/input \
        --output-dir /path/to/data/output \
        --metadata-path /path/to/data/input/render_metadata.json \
        --render-curriculum 1 \
        --checkpoint checkpoints/best_model.pth
"""

from pathlib import Path
import argparse
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import MultiViewPBRGenerator
from train_config import (
    TrainConfig,
    get_default_config,
    get_quick_test_config,
    get_lightweight_config,
)
from utils.dataset import PBRDataset


def _load_config(config_arg: str) -> TrainConfig:
    """Mirror train.py's config loading logic."""
    if config_arg == "default":
        config = get_default_config()
    elif config_arg == "quick_test":
        config = get_quick_test_config()
    elif config_arg == "lightweight":
        config = get_lightweight_config()
    else:
        # Custom config file path
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "custom_config", config_arg)
        custom_config = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(custom_config)
        config = custom_config.get_config()
    return config


def _apply_data_overrides(config: TrainConfig, args: argparse.Namespace) -> None:
    """Apply CLI overrides for data paths and dirty/clean choice."""
    root_path = Path(args.data_root) if args.data_root else None

    # Input/output/metadata: explicit CLI wins; otherwise fall back to data_root
    if args.input_dir:
        config.data.input_dir = args.input_dir
    elif config.data.input_dir is None and root_path:
        config.data.input_dir = str(root_path / "input")

    if args.output_dir:
        config.data.output_dir = args.output_dir
    elif config.data.output_dir is None and root_path:
        config.data.output_dir = str(root_path / "output")

    if args.metadata_path:
        config.data.metadata_path = args.metadata_path
    elif config.data.metadata_path is None and config.data.input_dir:
        config.data.metadata_path = str(
            Path(config.data.input_dir) / "render_metadata.json")

    if args.render_curriculum is not None:
        config.data.render_curriculum = args.render_curriculum
    elif args.use_dirty:
        config.data.render_curriculum = 2
    config.data.use_dirty_renders = (config.data.render_curriculum == 2)


def save_shard(out_dir: Path, idx: int, sample_indices, outputs):
    """Save a single shard of teacher outputs to disk (inputs/targets loaded from PNGs during training)."""
    shard_path = out_dir / f"shard_{idx:05d}.pt"
    
    # Stack lists into tensors and cast to float16
    # We ONLY save the teacher outputs to save massive amounts of disk space.
    # Inputs and Targets will be loaded from the original PNG dataset during student training.
    tensor_outputs = {k: torch.cat(v, dim=0).half() for k, v in outputs.items()}  # (B, C, H, W)
    
    torch.save(
        {
            "indices": sample_indices,
            "teacher_outputs": tensor_outputs,
        },
        shard_path,
    )
    print(f"Saved {shard_path}")


def run_inference(
    config: TrainConfig,
    checkpoint_path: str,
    out_dir: str = "teacher_shards",
    shard_size: int = 256,
    batch_size: int = 4,
    shard_output_size: int = None,
):
    """
    Run teacher inference and save shards.
    
    Args:
        config: Training config
        checkpoint_path: Path to teacher checkpoint
        out_dir: Output directory for shards
        shard_size: Number of samples per shard
        batch_size: Inference batch size
        shard_output_size: If set, downsample teacher outputs to this size (e.g., 1024).
                          This saves disk space and matches student output resolution.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1) Load teacher model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewPBRGenerator(config).to(device)
    
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    try:
        # PyTorch 2.6+ defaults to weights_only=True, which breaks custom configs
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only argument (and default to False)
        ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = ckpt["generator_state_dict"]
    
    # Sanitize state_dict keys (remove _orig_mod. or module. prefixes from torch.compile/DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("_orig_mod."):
            name = name[10:]
        elif name.startswith("module."):
            name = name[7:]
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Optimize model for inference
    if device.type == "cuda":
        # Use TF32 on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Compile model if possible (PyTorch 2.0+)
        # Note: Disabled by default due to instability (CUDA illegal memory access) with max-autotune on some setups
        if hasattr(torch, "compile") and False:  # Set to True to enable compilation
            print("Compiling model for faster inference...")
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}), proceeding without compilation.")

    # 2) Build dataset (no train/val split, use all samples once)
    if config.transform.use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = config.transform.mean
        std = config.transform.std

    ds = PBRDataset(
        input_dir=config.data.input_dir,
        output_dir=config.data.output_dir,
        metadata_path=config.data.metadata_path,
        transform_mean=mean,
        transform_std=std,
        image_size=config.data.image_size,
        use_dirty=config.data.use_dirty_renders,
        curriculum_mode=config.data.render_curriculum,
        split=None,  # use all samples
        val_ratio=config.data.val_ratio,
        seed=config.training.seed,
    )

    # Use DataLoader for parallel loading (significantly speeds up PNG decoding)
    # Reduced workers and disabled pin_memory to avoid CUDA/multiprocessing errors
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2
    )

    shard_idx = 0
    shard_samples = []
    # shard_inputs = []  <-- No longer saving inputs/targets to save disk space
    # shard_targets = [] 
    shard_outputs = {"albedo": [], "roughness": [],
                     "metallic": [], "normal": []}

    print(f"Running teacher over {len(ds)} samples (batch_size={batch_size})...")
    
    # Use AMP for faster inference on modern GPUs
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
              torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext())

    with torch.no_grad():
        for batch_i, (inputs_batch, targets_batch) in enumerate(tqdm(loader)):
            # inputs_batch: (B, 3, 3, H, W)
            # targets_batch: (B, 4, 3, H, W)
            current_batch_size = inputs_batch.shape[0]
            
            # Move to device
            inputs_device = inputs_batch.to(device, non_blocking=True)

            # Run teacher with AMP
            with amp_ctx:
                pred = model(inputs_device)
            
            # Process each sample in the batch
            for i in range(current_batch_size):
                # Calculate global sample index
                global_idx = batch_i * batch_size + i
                
                # Store data (move to CPU to save memory)
                shard_samples.append(global_idx)
                
                # We don't store inputs/targets anymore
                # shard_inputs.append(inputs_batch[i].unsqueeze(0)) 
                # shard_targets.append(targets_batch[i].unsqueeze(0))
                
                for k in shard_outputs:
                    output_tensor = pred[k][i].unsqueeze(0)
                    
                    # Downsample to shard_output_size if specified
                    if shard_output_size is not None and output_tensor.shape[-1] != shard_output_size:
                        output_tensor = F.interpolate(
                            output_tensor,
                            size=(shard_output_size, shard_output_size),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    shard_outputs[k].append(output_tensor.cpu())

                # When shard is full, write to disk
                if len(shard_samples) == shard_size:
                    save_shard(
                        out_dir_path, 
                        shard_idx,
                        shard_samples, 
                        shard_outputs
                    )
                    shard_idx += 1
                    shard_samples = []
                    # shard_inputs/targets are no longer collected
                    shard_outputs = {k: [] for k in shard_outputs}
                    
                    # Clear cache to prevent fragmentation over long runs
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        # Flush last partial shard
        if shard_samples:
            save_shard(
                out_dir_path, 
                shard_idx, 
                shard_samples, 
                shard_outputs
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run teacher model to generate distillation shards.")

    # Config (mirror train.py)
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config to use: 'default', 'quick_test', 'lightweight', or path to custom config",
    )

    # Data (same flags as train.py for familiarity)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing input/output subfolders (optional when using explicit directories)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing rendered samples (expects clean/ and optional dirty/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory containing ground-truth material folders",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to render_metadata.json mapping sample folders to materials",
    )
    parser.add_argument(
        "--use-dirty",
        action="store_true",
        help="Use dirty renders instead of the default clean renders",
    )
    parser.add_argument(
        "--render-curriculum",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="0=clean only, 1=match dataset clean/dirty ratio, 2=dirty only (overrides --use-dirty)",
    )

    # Distillation output
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained teacher checkpoint (e.g. checkpoints/best_model.pth).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="teacher_shards",
        help="Directory to save .pt shard files.",
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default=None,
        help="Alias for --out-dir (Directory to save .pt shard files).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=8,
        help="Number of samples per .pt shard. Default 8 (~4GB at 2048x2048).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size. Default 4.",
    )
    parser.add_argument(
        "--shard-output-size",
        type=int,
        default=None,
        help="Downsample teacher outputs to this size before saving (e.g., 1024 for 512 student with SR 2x).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Ensure safe multiprocessing start method for CUDA
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = parse_args()
    
    # 1. Load base config (default or from file)
    cfg = _load_config(args.config)
    
    # 2. Try to load config from checkpoint to ensure architecture matches weights
    # This prevents "size mismatch" errors if the user forgets --config
    try:
        print(f"Inspecting checkpoint: {args.checkpoint}")
        # Load on CPU to avoid OOM before we're ready
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            print("  Found config in checkpoint. Using it to ensure model architecture matches.")
            saved_config = ckpt["config"]
            # Use the saved config as the base, but we will override data paths below
            cfg = saved_config
        else:
            print("  No config found in checkpoint. Using CLI/default config.")
    except Exception as e:
        print(f"  Warning: Could not load config from checkpoint ({e}).")
        print("  Proceeding with CLI/default config.")

    # 3. Apply CLI overrides (input/output dirs, curriculum)
    _apply_data_overrides(cfg, args)
    
    # Handle alias
    out_dir = args.shards_dir if args.shards_dir else args.out_dir
    
    run_inference(
        config=cfg,
        checkpoint_path=args.checkpoint,
        out_dir=out_dir,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        shard_output_size=args.shard_output_size,
    )
