import argparse, pickle, torch
from pathlib import Path
from train import MultiViewPBRGenerator
from train_config import get_default_config, TrainConfig
from utils.dataset import PBRDataset
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--sample-idx", type=int, default=0)
parser.add_argument("--out-dir", default="inference_outputs")
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

ds = PBRDataset(
    cfg.data.input_dir, cfg.data.output_dir, cfg.data.metadata_path,
    cfg.transform.mean, cfg.transform.std, cfg.data.image_size,
    cfg.data.use_dirty_renders, split=None, val_ratio=cfg.data.val_ratio,
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
save_image(pred["normal"], out_dir / "normal.png", normalize=True, value_range=(0, 1))
save_image(pred["roughness"], out_dir / "roughness.png")
save_image(pred["metallic"], out_dir / "metallic.png")

# Also save the normalized input renders actually fed to the model
mean = torch.tensor(cfg.transform.mean, dtype=inputs_cpu.dtype).view(1, 3, 1, 1)
std = torch.tensor(cfg.transform.std, dtype=inputs_cpu.dtype).view(1, 3, 1, 1)
denorm_inputs = inputs_cpu * std + mean
denorm_inputs = denorm_inputs.clamp(0.0, 1.0)
for view_idx, view in enumerate(denorm_inputs):
    save_image(view, out_dir / f"view_{view_idx}.png")