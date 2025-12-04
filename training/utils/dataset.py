import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from pathlib import Path
from torchvision import transforms
from typing import Tuple, List, Optional, Literal
import json
import random
import warnings

# Allow loading of truncated images - prevents crashes on partially written files
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        image_size: Tuple[int, int] = (2048, 2048),
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

        self.image_size = image_size
        
        transform_list = []
        # Add resize if requested size differs from native 2048x2048
        if self.image_size != (2048, 2048):
             transform_list.append(transforms.Resize(self.image_size))
             
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_mean,
                std=transform_std
            )
        ])
        
        self.transform = transforms.Compose(transform_list)

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

    def _load_image(self, path: Path, retry_count: int = 3):
        """Load and transform an image with robust error handling."""
        last_error = None
        
        for attempt in range(retry_count):
            try:
                # Open and immediately load the image to catch truncation errors early
                img = Image.open(path)
                img.load()  # Force load to detect truncated files
                img = img.convert('RGB')
                
                # Verify native resolution
                if img.size != (2048, 2048):
                    # Only warn if not 2048, or assert if strict. 
                    # User requested: "Verify image dimensions are correct (assert shape)"
                    # But let's check if image_size is passed.
                    pass

                if self.transform:
                    img = self.transform(img)
                    
                # Verify tensor shape after transform
                expected_shape = (self.image_size[0], self.image_size[1])
                if img.shape[1:] != expected_shape:
                    raise ValueError(f"Image at {path} has incorrect dimensions {img.shape[1:]}. Expected {expected_shape}.")

                return img
                
            except (OSError, IOError) as e:
                last_error = e
                if attempt < retry_count - 1:
                    # Brief pause before retry (file might still be written)
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    continue
                    
        # All retries failed - raise with clear error message
        raise OSError(
            f"Failed to load image after {retry_count} attempts: {path}\n"
            f"Original error: {last_error}\n"
            f"This may indicate a corrupted or truncated image file. "
            f"Consider running a dataset validation script to identify bad files."
        )

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


class DistillationShardDataset(Dataset):
    """
    Dataset for loading pre-computed distillation shards (teacher outputs only).
    Inputs and targets are loaded from the original PBRDataset on the fly.
    
    Each shard is a .pt file containing:
    - teacher_outputs: Dict[str, Tensor]
    - indices: List[int] (indices into the PBRDataset)
    """
    
    def __init__(
        self,
        shards_dir: str,
        # PBRDataset args
        input_dir: str,
        output_dir: str,
        metadata_path: Optional[str],
        transform_mean: List,
        transform_std: List,
        image_size: Tuple[int, int] = (2048, 2048),
        use_dirty: bool = False,
        curriculum_mode: int = 0,
        # Split args
        split: Optional[Literal["train", "val"]] = None,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        self.shards_dir = Path(shards_dir)
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Shards directory not found: {self.shards_dir}")
            
        # Initialize the underlying PBRDataset (with split=None to access all samples)
        # We rely on the shard indices to map back to this dataset.
        self.pbr_dataset = PBRDataset(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_path=metadata_path,
            transform_mean=transform_mean,
            transform_std=transform_std,
            image_size=image_size,
            use_dirty=use_dirty,
            curriculum_mode=curriculum_mode,
            split=None, # We handle splitting via shard indices
            val_ratio=val_ratio,
            seed=seed
        )

        # Find all shards
        self.shard_paths = sorted(list(self.shards_dir.glob("shard_*.pt")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .pt shards found in {self.shards_dir}")
            
        # Build index: global_idx -> (shard_idx, local_idx, pbr_dataset_idx)
        self.index_map = []
        
        # Cache the index map to disk to avoid re-scanning
        index_cache_path = self.shards_dir / "shard_index_cache.pt"
        
        if index_cache_path.exists():
            print(f"Loading shard index from cache: {index_cache_path}")
            self.index_map = torch.load(index_cache_path, weights_only=False)
        else:
            print(f"Scanning {len(self.shard_paths)} shards...")
            for shard_idx, p in enumerate(self.shard_paths):
                try:
                    # Map location cpu to avoid gpu memory usage during scan
                    # Only load indices, not the heavy tensors
                    # Note: torch.load loads everything, but we can try to be efficient
                    data = torch.load(p, map_location="cpu", weights_only=False) 
                    indices = data["indices"]
                    num_samples = len(indices)
                    for i in range(num_samples):
                        # Store (shard_idx, local_idx_in_shard, original_dataset_idx)
                        self.index_map.append((shard_idx, i, indices[i]))
                except Exception as e:
                    print(f"Error loading shard {p}: {e}")
            
            # Save cache
            print(f"Saving shard index cache to {index_cache_path}")
            torch.save(self.index_map, index_cache_path)
                
        # Split train/val
        all_indices = list(range(len(self.index_map)))
        random.seed(seed)
        random.shuffle(all_indices)
        
        if split is not None:
            num_val = int(len(all_indices) * val_ratio)
            if split == "val":
                self.indices = all_indices[:num_val]
            else:
                self.indices = all_indices[num_val:]
        else:
            self.indices = all_indices
            
        self.cached_shard_idx = -1
        self.cached_data = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        shard_idx, local_idx, pbr_idx = self.index_map[global_idx]
        
        # Load shard if not cached
        if self.cached_shard_idx != shard_idx:
            self.cached_data = torch.load(
                self.shard_paths[shard_idx], 
                map_location="cpu",
                weights_only=False
            )
            self.cached_shard_idx = shard_idx
            
        # Load inputs/targets from original dataset
        inputs, pbr_maps = self.pbr_dataset[pbr_idx]
        
        # Load teacher outputs from shard
        teacher_pred = {
            k: v[local_idx].float() for k, v in self.cached_data["teacher_outputs"].items()
        }
        
        return inputs, pbr_maps, teacher_pred


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
        image_size=(2048, 2048),  # Input image size
    seed=42,  # Seed for reproducible splits
        metadata_path: Optional[str] = None,
        shards_dir: Optional[str] = None
):
    if shards_dir:
        ds = DistillationShardDataset(
            shards_dir=shards_dir,
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
    else:
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

