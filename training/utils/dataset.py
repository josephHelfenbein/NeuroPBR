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
        ├── clean/sample_XXXX/{0,1,2}.png  (used by curriculum 0/1)
        ├── dirty/sample_XXXX/{0,1,2}.png  (used by curriculum 1/2)
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

    The curriculum is controlled via `curriculum_mode`:
        0 -> clean renders only
        1 -> match on-disk clean/dirty proportions (each sample listed once per source)
        2 -> dirty renders only
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
        curriculum_mode: int = 0,
        split: Optional[Literal["train", "val"]] = None,
        val_ratio: float = 0.1,
        seed: int = 42  # For reproducible splits
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_dirty = use_dirty
        self.curriculum_mode = curriculum_mode if curriculum_mode is not None else 0
        if self.curriculum_mode not in (0, 1, 2):
            raise ValueError("curriculum_mode must be 0 (clean), 1 (mixed), or 2 (dirty)")
        self.metadata_path = Path(metadata_path) if metadata_path else self.input_dir / 'render_metadata.json'

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        # Resolve render directories and availability
        self.clean_dir = self.input_dir / 'clean'
        self.dirty_dir = self.input_dir / 'dirty'
        if self.curriculum_mode in (0, 1) and not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean render directory not found: {self.clean_dir}")
        if self.curriculum_mode in (1, 2) and not self.dirty_dir.exists():
            raise FileNotFoundError(f"Dirty render directory not found: {self.dirty_dir}")

        self.transform = transforms.Compose([
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
        
        # Build combined sample list based on curriculum
        all_samples = self._build_curriculum_samples()
        
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

    def _load_samples_from_dir(self, render_dir: Path):
        """Load valid samples for a specific render directory."""
        samples = []
        for sample_folder in sorted(render_dir.glob('sample_*')):
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

    def _build_curriculum_samples(self):
        """Create the master sample list (with source tags) for the chosen curriculum."""
        clean_samples = self._load_samples_from_dir(self.clean_dir) if self.clean_dir.exists() else []
        dirty_samples = self._load_samples_from_dir(self.dirty_dir) if self.dirty_dir.exists() else []

        tagged = []
        if self.curriculum_mode == 0:
            if not clean_samples:
                raise ValueError("Curriculum 0 requires clean renders but none were found.")
            tagged = [(name, mat, 'clean') for name, mat in clean_samples]
        elif self.curriculum_mode == 2:
            if not dirty_samples:
                raise ValueError("Curriculum 2 requires dirty renders but none were found.")
            tagged = [(name, mat, 'dirty') for name, mat in dirty_samples]
        else:  # mixed, mirror on-disk proportions
            if not clean_samples or not dirty_samples:
                raise ValueError("Curriculum 1 requires both clean and dirty renders to be present.")
            clean_dict = {name: mat for name, mat in clean_samples}
            dirty_dict = {name: mat for name, mat in dirty_samples}
            all_names = sorted(set(clean_dict.keys()) | set(dirty_dict.keys()))
            for name in all_names:
                if name in clean_dict:
                    tagged.append((name, clean_dict[name], 'clean'))
                if name in dirty_dict:
                    tagged.append((name, dirty_dict[name], 'dirty'))

        return tagged

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path):
        """Load and transform an image."""
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: int):
        sample_name, material_name, source = self.samples[idx]

        # Select render directory per sample source
        if source == 'dirty':
            render_dir = self.dirty_dir
        else:
            render_dir = self.clean_dir

        # Load 3 input renders (clean or dirty)
        input_renders = torch.stack([
            self._load_image(render_dir / sample_name / f)
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
        use_dirty=False,  # Deprecated, prefer curriculum_mode
        curriculum_mode: int = 0,
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
        curriculum_mode=curriculum_mode,
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

