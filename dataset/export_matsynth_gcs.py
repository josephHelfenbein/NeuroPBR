from datasets import load_dataset
from PIL import Image
from google.cloud import storage
import io
import json

import argparse

def main(bucket_name="main-testing", prefix="raw", limit=4000, start=0):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if start > limit:
        raise ValueError("start must be less than limit")

    if start > 0:
        print(f"Starting from index {start}")
    
    ds = load_dataset("gvecchio/MatSynth", streaming=True, split="train")
    
    i = 0
    image_keys = ["basecolor", "normal", "roughness", "metallic",
                  "diffuse", "specular", "displacement", "opacity", "blend_mask"]
    
    for ex in ds:
        if not any(ex.get(k) for k in ("basecolor", "normal", "roughness", "metallic")):
            continue
        
        if i < start:
            i += 1
            continue
        
        # Upload images directly to GCS
        for key in image_keys:
            if ex.get(key) is not None:
                buf = io.BytesIO()
                img = ex[key] if isinstance(ex[key], Image.Image) else Image.fromarray(ex[key])
                img.save(buf, format='PNG')
                buf.seek(0)
                
                # Upload directly
                blob = bucket.blob(f"{prefix}/mat_{i:05d}/{key}.png")
                blob.upload_from_file(buf, content_type='image/png')
        
        i += 1
        if i % 10 == 0:
            print(f"Uploaded {i}/{limit} materials")
        
        if i >= limit:
            break
    
    print(f"Done! Exported {i} materials")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MatSynth dataset to Google Cloud Storage.")
    parser.add_argument("--bucket", type=str, default="main-testing", help="GCS bucket name")
    parser.add_argument("--prefix", type=str, default="raw", help="Prefix for uploaded files")
    parser.add_argument("--limit", type=int, default=4000, help="Maximum number of materials to export")
    parser.add_argument("--start", type=int, default=0, help="Start index to resume export")
    
    args = parser.parse_args()
    
    main(
        bucket_name=args.bucket,
        prefix=args.prefix,
        limit=args.limit,
        start=args.start
    )