"""
Student Model Inference Script for NeuroPBR

Runs inference with a trained student model (lightweight MobileNetV3-based).
The student model outputs PBR maps from 3 input rendered views.

This script is identical to the teacher inference except it loads student checkpoints.

Usage:
    # Run inference on a specific sample from the dataset
    python student/run_inference.py \
        --checkpoint checkpoints/best_student.pth \
        --sample-idx 0 \
        --out-dir student_outputs

    # Run inference on custom input images
    python student/run_inference.py \
        --checkpoint checkpoints/best_student.pth \
        --input-dir /path/to/renders \
        --out-dir student_outputs

    # Compare student vs teacher outputs
    python student/run_inference.py \
        --checkpoint checkpoints/best_student.pth \
        --input-dir /path/to/renders \
        --out-dir student_outputs
"""

import sys
import argparse
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from train import MultiViewPBRGenerator
from train_config import get_default_config, TrainConfig
from utils.dataset import PBRDataset


def _normalize_image_size(image_size):
    """Normalize image size to (height, width) tuple."""
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2:
            raise ValueError("image_size must contain [height, width]")
        return int(image_size[0]), int(image_size[1])
    size = int(image_size)
    return size, size


def _build_input_transform(image_size, mean, std):
    """Build input transformation pipeline."""
    height, width = _normalize_image_size(image_size)
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _load_inputs_from_directory(directory: Path, transform):
    """
    Load exactly 3 PNG images from a directory.

    Args:
        directory: Path to directory containing 3 PNG renders
        transform: Transformation to apply to each image

    Returns:
        Stacked tensor of shape (3, C, H, W)
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    # Find all PNG files and sort them
    image_paths = sorted(p for p in directory.glob("*.png") if p.is_file())

    if len(image_paths) != 3:
        raise ValueError(
            f"Expected exactly 3 PNG renders in {directory}, found {len(image_paths)}\n"
            f"Files found: {[p.name for p in image_paths]}"
        )

    # Load and transform each image
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))

    return torch.stack(tensors)


def main():
    parser = argparse.ArgumentParser(description="Student Model Inference")

    # Model checkpoint
    parser.add_argument("--checkpoint", required=True,
                      help="Path to student model checkpoint")

    # Input selection (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--sample-idx", type=int, default=None,
                           help="Sample index from dataset (default: 0)")
    input_group.add_argument("--input-dir", type=str, default=None,
                           help="Directory containing exactly three PNG renders")

    # Output
    parser.add_argument("--out-dir", default="student_inference_outputs",
                      help="Output directory for generated PBR maps")

    # Device
    parser.add_argument("--device", type=str, default=None,
                      help="Device: 'cuda', 'cpu', or None (auto-detect)")

    args = parser.parse_args()

    # Set default sample index if neither input method specified
    if args.sample_idx is None and args.input_dir is None:
        args.sample_idx = 0

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading student checkpoint: {args.checkpoint}")
    try:
        with torch.serialization.safe_globals([TrainConfig]):
            ckpt = torch.load(args.checkpoint, map_location=device)
    except pickle.UnpicklingError:
        # Fallback for trusted local checkpoints
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get config (try student-specific key first, fallback to generator)
    cfg = ckpt.get("config", get_default_config())

    # Build student model
    print("Building student model...")
    model = MultiViewPBRGenerator(cfg).to(device)

    # Load student weights (try student_state_dict first, fallback to generator_state_dict)
    if "student_state_dict" in ckpt:
        model.load_state_dict(ckpt["student_state_dict"])
        print("Loaded student_state_dict")
    elif "generator_state_dict" in ckpt:
        model.load_state_dict(ckpt["generator_state_dict"])
        print("Loaded generator_state_dict (generic)")
    else:
        raise KeyError("Checkpoint does not contain 'student_state_dict' or 'generator_state_dict'")

    model.eval()
    print(f"Student model loaded successfully")
    print(f"  Encoder: {cfg.model.encoder_type} ({cfg.model.encoder_backbone})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build input transform
    if hasattr(cfg, 'transform'):
        mean = cfg.transform.mean
        std = cfg.transform.std
    else:
        # Fallback defaults
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transform = _build_input_transform(cfg.data.image_size, mean, std)

    # Load input images
    if args.input_dir:
        print(f"Loading inputs from directory: {args.input_dir}")
        inputs_cpu = _load_inputs_from_directory(Path(args.input_dir), transform)
    else:
        print(f"Loading sample {args.sample_idx} from dataset...")
        curriculum = getattr(cfg.data, "render_curriculum", 2 if cfg.data.use_dirty_renders else 0)
        ds = PBRDataset(
            cfg.data.input_dir,
            cfg.data.output_dir,
            cfg.data.metadata_path,
            mean,
            std,
            cfg.data.image_size,
            cfg.data.use_dirty_renders,
            curriculum_mode=curriculum,
            split=None,
            val_ratio=cfg.data.val_ratio,
            seed=cfg.training.seed,
        )
        inputs_cpu, _ = ds[args.sample_idx]

    # Move to device and add batch dimension
    inputs = inputs_cpu.unsqueeze(0).to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        pred = model(inputs)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save PBR outputs
    print(f"Saving outputs to {out_dir}")
    save_image(pred["albedo"], out_dir / "albedo.png", normalize=True, value_range=(0, 1))
    save_image(pred["normal"], out_dir / "normal.png", normalize=True, value_range=(0, 1))
    save_image(pred["roughness"], out_dir / "roughness.png")
    save_image(pred["metallic"], out_dir / "metallic.png")

    # Save denormalized input renders for reference
    mean_tensor = torch.tensor(mean, dtype=inputs_cpu.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=inputs_cpu.dtype).view(1, 3, 1, 1)
    denorm_inputs = inputs_cpu * std_tensor + mean_tensor
    denorm_inputs = denorm_inputs.clamp(0.0, 1.0)

    for view_idx, view in enumerate(denorm_inputs):
        save_image(view, out_dir / f"input_view_{view_idx}.png")

    print("Done!")
    print(f"\nOutputs saved:")
    print(f"  - {out_dir / 'albedo.png'}")
    print(f"  - {out_dir / 'roughness.png'}")
    print(f"  - {out_dir / 'metallic.png'}")
    print(f"  - {out_dir / 'normal.png'}")
    print(f"  - {out_dir / 'input_view_*.png'} (reference inputs)")


if __name__ == "__main__":
    main()
