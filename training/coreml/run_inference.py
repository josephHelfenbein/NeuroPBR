import argparse
import coremltools as ct
from pathlib import Path
from PIL import Image
import os

def load_inputs(input_dir, size=(2048, 2048)):
    """
    Loads 3 images from the directory, sorts them, and resizes them.
    Returns a dictionary suitable for the Core ML model.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    # Find images
    extensions = {".png", ".jpg", ".jpeg"}
    images = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in extensions])
    
    if len(images) < 3:
        raise ValueError(f"Found {len(images)} images in {input_dir}. Expected at least 3.")
    
    # Take first 3
    selected_images = images[:3]
    print(f"Using images: {[f.name for f in selected_images]}")
    
    input_dict = {}
    for i, img_path in enumerate(selected_images):
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.Resampling.BILINEAR)
        input_dict[f"view{i+1}"] = img
        
    return input_dict

def run_inference(model_path, input_dir, output_dir):
    print(f"Loading model: {model_path}")
    model = ct.models.MLModel(model_path)
    
    # Model expects 512x512 input (memory-optimized for iPhone)
    # Outputs will be upscaled to 2048x2048
    input_size = (512, 512)
    output_size = (2048, 2048)
    
    print(f"Preparing inputs (resizing to {input_size[0]}x{input_size[1]})...")
    inputs = load_inputs(input_dir, size=input_size)
    
    print("Running prediction...")
    outputs = model.predict(inputs)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving outputs to {output_dir} (upscaling to {output_size[0]}x{output_size[1]})...")
    for key, value in outputs.items():
        if isinstance(value, Image.Image):
            out_path = output_dir / f"{key}.png"
            
            # Upscale to 2048x2048 using Lanczos resampling (matches iOS app behavior)
            if value.size != output_size:
                value = value.resize(output_size, Image.Resampling.LANCZOS)
            
            # Note on Normal Map:
            # The model output for 'normal' is configured with scale=0.5, bias=0.5.
            # This maps the internal [-1, 1] range to [0, 1] in the output image.
            
            value.save(out_path)
            print(f"Saved {out_path}")
        else:
            print(f"Warning: Output '{key}' is not an image ({type(value)}). Skipping save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with NeuroPBR Core ML model")
    parser.add_argument("model", type=str, help="Path to .mlpackage")
    parser.add_argument("--input", type=str, required=True, help="Directory containing 3 input images")
    parser.add_argument("--output", type=str, default="coreml_output", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    run_inference(args.model, args.input, args.output)
