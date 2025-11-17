import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Tuple, List, Optional, Literal
import json
import random

class PBRDataset(Dataset):
    """
    NeuroPBR Dataset
    
    Required layout (configurable via explicit directories):
        input_dir/
        ├── clean/sample_XXXX/{0,1,2}.png  (required default inputs)
        ├── dirty/sample_XXXX/{0,1,2}.png  (optional, used when use_dirty=True)
        └── render_metadata.json  (maps sample_XXXX -> material_name)
    
        output_dir/
        └── material_name/
            ├── albedo.png
            ├── roughness.png
            ├── metallic.png
            └── normal.png
    
    Returns:
        input_renders: (3, 3, H, W) - 3 RGB renders (clean or dirty)
        pbr_maps: (4, 3, H, W) - 4 PBR maps (albedo, roughness, metallic, normal)
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        metadata_path: Optional[str],
        transform_mean: List,
        transform_std: List,
        image_size: Tuple[int, int] = (1024, 1024),
        use_dirty: bool = False,
        split: Optional[Literal["train", "val"]] = None,
        val_ratio: float = 0.1,
        seed: int = 42  # For reproducible splits
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_dirty = use_dirty
        self.metadata_path = Path(metadata_path) if metadata_path else self.input_dir / 'render_metadata.json'

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Choose clean or dirty renders
        self.render_dir = self.input_dir / ('dirty' if use_dirty else 'clean')
        if not self.render_dir.exists():
            raise FileNotFoundError(f"Render directory not found: {self.render_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_mean,
                std=transform_std
            )
        ])

        self.render_files = ['0.png', '1.png', '2.png']
        self.pbr_files = ['albedo.png', 'roughness.png', 'metallic.png', 'normal.png']

        # Load metadata mapping
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Get all samples and shuffle for representative split
        all_samples = self._load_samples()
        
        # Shuffle samples for random train/val split (reproducible with seed)
        random.seed(seed)
        random.shuffle(all_samples)
        
        # Split train/val if specified
        if split is not None:
            num_val = int(len(all_samples) * val_ratio)
            if split == "val":
                self.samples = all_samples[:num_val]
            else:  # train
                self.samples = all_samples[num_val:]
        else:
            self.samples = all_samples

    def _load_samples(self):
        """Load all valid samples that have both input renders and output PBR maps."""
        samples = []
        for sample_folder in sorted(self.render_dir.glob('sample_*')):
            sample_name = sample_folder.name
            
            # Check if sample has metadata mapping
            if sample_name not in self.metadata:
                continue
            
            material_name = self.metadata[sample_name]
            material_dir = self.output_dir / material_name
            
            # Check if all required files exist
            has_all_renders = all(
                (sample_folder / f).exists() for f in self.render_files
            )
            has_all_pbr = all(
                (material_dir / f).exists() for f in self.pbr_files
            )
            
            if has_all_renders and has_all_pbr:
                samples.append((sample_name, material_name))
        
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path):
        """Load and transform an image."""
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: int):
        sample_name, material_name = self.samples[idx]

        # Load 3 input renders (clean or dirty)
        input_renders = torch.stack([
            self._load_image(self.render_dir / sample_name / f)
            for f in self.render_files
        ])  # [3, C, H, W]

        # Load 4 ground truth PBR maps
        pbr_maps = torch.stack([
            self._load_image(self.output_dir / material_name / f)
            for f in self.pbr_files
        ])  # [4, C, H, W]

        return input_renders, pbr_maps

'''
batch_size=  # Per GPU! Total = 16 * 8 = 128
num_workers= # 8 workers per GPU = 64 workers total
pin_memory=True, # Faster GPU transfer    # just uses some more ram
persistent_workers=True  # Keep workers alive between epochs     # just uses some more ram
prefetch_factor=2 # each worker works on loading 2 batches
sampler is for multi gpu training
'''
def get_dataloader(
    input_dir: str,
    output_dir: str,
    transform_mean,  # load from config
    transform_std,  # load from config
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        sampler=None,
        use_dirty=False,  # Use dirty or clean renders
        split=None,  # "train" or "val" or None
        val_ratio=0.1,  # Validation split ratio
        image_size=(1024, 1024),  # Input image size
    seed=42,  # Seed for reproducible splits
        metadata_path: Optional[str] = None
):
    ds = PBRDataset(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_path=metadata_path,
        transform_mean=transform_mean,
        transform_std=transform_std,
        image_size=image_size,
        use_dirty=use_dirty,
        split=split,
        val_ratio=val_ratio,
        seed=seed
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2
    )

