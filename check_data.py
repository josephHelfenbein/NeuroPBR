import torch
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

def check_shards(shards_dir, num_check=10):
    print(f"Checking first {num_check} shards in {shards_dir}...")
    shards = sorted(list(Path(shards_dir).glob("shard_*.pt")))
    
    if not shards:
        print("No shards found!")
        return

    for i, shard_path in enumerate(shards[:num_check]):
        try:
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
            teacher_outputs = data["teacher_outputs"]
            
            has_nan = False
            for k, v in teacher_outputs.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"❌ Shard {shard_path.name}: NaN/Inf found in key '{k}'")
                    has_nan = True
                    break
            
            if not has_nan:
                print(f"✅ Shard {shard_path.name}: OK")
                
        except Exception as e:
            print(f"❌ Shard {shard_path.name}: Error loading - {e}")

def check_images(input_dir, num_check=20):
    print(f"\nChecking first {num_check} images in {input_dir}...")
    input_path = Path(input_dir)
    
    # Check clean/dirty folders
    images = list(input_path.glob("**/*.png"))
    
    if not images:
        print("No images found!")
        return

    for i, img_path in enumerate(images[:num_check]):
        try:
            img = Image.open(img_path)
            img_data = np.array(img)
            if np.isnan(img_data).any():
                print(f"❌ Image {img_path.name}: NaN found")
            else:
                # print(f"✅ Image {img_path.name}: OK")
                pass
        except Exception as e:
            print(f"❌ Image {img_path.name}: Error loading - {e}")
    print("Image check complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    args = parser.parse_args()
    
    check_shards(args.shards_dir)
    check_images(args.input_dir)
