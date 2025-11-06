import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Tuple, List

class PBRDataset(Dataset):
    """
    Simple dataset: returns individual images, model decides how to use them
    """

    def __init__(self, root_dir: str, transform_mean: List, transform_std: List, image_size: Tuple[int, int]=(2048,2048)):
        self.root_dir = Path(root_dir)
        self.clean_dir = self.root_dir / 'clean'
        self.dirty_dir = self.root_dir / 'dirty'
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_mean,
                std=transform_std
            )
        ])

        self.samples = self._load_samples()
        self.dirty_files = ['0.png', '1.png', '2.png']
        self.clean_files = ['albedo.png', 'roughness.png', 'metallic.png', 'normal.png']

    def _load_samples(self):
        samples = []
        for sample_folder in sorted(self.clean_dir.glob('sample_*')):
            sample_name = sample_folder.name
            if (self.dirty_dir / sample_name).exists():
                samples.append(sample_name)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str):
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: int):
        sample_name = self.samples[idx]

        # Load 3 dirty images
        dirty = torch.stack([
            self._load_image(self.dirty_dir / sample_name / f)
            for f in self.dirty_files
        ])  # [3, C, H, W]

        # Load 4 clean PBR textures
        clean = torch.stack([
            self._load_image(self.clean_dir / sample_name / f)
            for f in self.clean_files
        ])  # [4, C, H, W]

        return dirty, clean

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
        transform_mean, # laod from config
        transform_std, # laod from config
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        sampler=None
):
    ds = PBRDataset(
        root_dir=root_dir,
        transform_mean=transform_mean,
        transform_std=transform_std,
        image_size=(2048, 2048)
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

