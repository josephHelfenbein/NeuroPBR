import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Tuple, List, Optional, Literal
import json

class PBRDataset(Dataset):
    """
    NeuroPBR Dataset
    
    Structure:
        root_dir/
        ├── input/
        │   ├── clean/
        │   │   └── sample_XXXX/
        │   │       ├── 0.png, 1.png, 2.png  (3 clean renders)
        │   ├── dirty/
        │   │   └── sample_XXXX/
        │   │       ├── 0.png, 1.png, 2.png  (3 dirty renders)
        │   └── render_metadata.json  (maps sample_XXXX -> material_name)
        └── output/
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
        root_dir: str, 
        transform_mean: List, 
        transform_std: List, 
        image_size: Tuple[int, int] = (1024, 1024),
        use_clean: bool = False,  # If True, use clean renders; if False, use dirty
        split: Optional[Literal["train", "val"]] = None,
        val_ratio: float = 0.1
    ):
        self.root_dir = Path(root_dir)
        self.input_dir = self.root_dir / 'input'
        self.output_dir = self.root_dir / 'output'
        self.use_clean = use_clean
        
        # Choose clean or dirty renders
        self.render_dir = self.input_dir / ('clean' if use_clean else 'dirty')
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_mean,
                std=transform_std
            )
        ])

        # Load metadata mapping
        metadata_path = self.input_dir / 'render_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Get all samples
        all_samples = self._load_samples()
        
        # Split train/val if specified
        if split is not None:
            num_val = int(len(all_samples) * val_ratio)
            if split == "val":
                self.samples = all_samples[:num_val]
            else:  # train
                self.samples = all_samples[num_val:]
        else:
            self.samples = all_samples
        
        self.render_files = ['0.png', '1.png', '2.png']
        self.pbr_files = ['albedo.png', 'roughness.png', 'metallic.png', 'normal.png']

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
        root_dir: str,
        transform_mean,  # load from config
        transform_std,  # load from config
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        sampler=None,
        use_clean=False,  # Use clean or dirty renders
        split=None,  # "train" or "val" or None
        val_ratio=0.1,  # Validation split ratio
        image_size=(1024, 1024)  # Input image size
):
    ds = PBRDataset(
        root_dir=root_dir,
        transform_mean=transform_mean,
        transform_std=transform_std,
        image_size=image_size,
        use_clean=use_clean,
        split=split,
        val_ratio=val_ratio
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

