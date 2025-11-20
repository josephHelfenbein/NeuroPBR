import argparse, pickle, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils.visualization import normal_to_rgb

from train import MultiViewPBRGenerator
from train_config import get_default_config, TrainConfig
from utils.dataset import PBRDataset


def _normalize_image_size(image_size):
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2:
            raise ValueError("image_size must contain [height, width]")
        return int(image_size[0]), int(image_size[1])
    size = int(image_size)
    return size, size


def _build_input_transform(image_size, mean, std):
    height, width = _normalize_image_size(image_size)
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _load_inputs_from_directory(directory: Path, transform):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    image_paths = sorted(p for p in directory.glob("*.png") if p.is_file())
    if len(image_paths) != 3:
        raise ValueError(f"Expected exactly 3 PNG renders in {directory}, found {len(image_paths)}")
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--sample-idx", type=int, default=0)
parser.add_argument("--out-dir", default="inference_outputs")
parser.add_argument("--input-dir", type=str, help="Directory containing exactly three PNG renders to use for inference.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with torch.serialization.safe_globals([TrainConfig]):
        ckpt = torch.load(args.checkpoint, map_location=device)
except pickle.UnpicklingError:
    # fall back to weights_only=False for trusted checkpoints created locally
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
cfg = ckpt.get("config", get_default_config())

model = MultiViewPBRGenerator(cfg).to(device)
model.load_state_dict(ckpt["generator_state_dict"])
model.eval()

transform = _build_input_transform(cfg.data.image_size, cfg.transform.mean, cfg.transform.std)

if args.input_dir:
    inputs_cpu = _load_inputs_from_directory(Path(args.input_dir), transform)
else:
    curriculum = getattr(cfg.data, "render_curriculum", 2 if cfg.data.use_dirty_renders else 0)
    ds = PBRDataset(
        cfg.data.input_dir, cfg.data.output_dir, cfg.data.metadata_path,
        cfg.transform.mean, cfg.transform.std, cfg.data.image_size,
        cfg.data.use_dirty_renders, curriculum_mode=curriculum,
        split=None, val_ratio=cfg.data.val_ratio,
        seed=cfg.training.seed,
    )
    inputs_cpu, _ = ds[args.sample_idx]
inputs = inputs_cpu.unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(inputs)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# Save generator outputs
save_image(pred["albedo"], out_dir / "albedo.png", normalize=True, value_range=(0, 1))

# Normal map prediction is in [-1, 1]; convert to displayable RGB before saving.
normal_rgb = normal_to_rgb(pred["normal"], normalize=True)
save_image(normal_rgb, out_dir / "normal.png", normalize=False)

save_image(pred["roughness"], out_dir / "roughness.png")
save_image(pred["metallic"], out_dir / "metallic.png")

# Also save the normalized input renders actually fed to the model
mean = torch.tensor(cfg.transform.mean, dtype=inputs_cpu.dtype).view(1, 3, 1, 1)
std = torch.tensor(cfg.transform.std, dtype=inputs_cpu.dtype).view(1, 3, 1, 1)
denorm_inputs = inputs_cpu * std + mean
denorm_inputs = denorm_inputs.clamp(0.0, 1.0)
for view_idx, view in enumerate(denorm_inputs):
    save_image(view, out_dir / f"view_{view_idx}.png")