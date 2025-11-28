# export_matsynth_direct.py
from datasets import load_dataset
from PIL import Image
from google.cloud import storage
import io
import json

def main(bucket_name="main-testing", prefix="raw", limit=4000):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    ds = load_dataset("gvecchio/MatSynth", streaming=True, split="train")
    
    i = 0
    image_keys = ["basecolor", "normal", "roughness", "metallic",
                  "diffuse", "specular", "displacement", "opacity", "blend_mask"]
    
    for ex in ds:
        if not any(ex.get(k) for k in ("basecolor", "normal", "roughness", "metallic")):
            continue
        
        # Upload images directly to GCS
        for key in image_keys:
            if ex.get(key) is not None:
                # Convert to bytes
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
    main()