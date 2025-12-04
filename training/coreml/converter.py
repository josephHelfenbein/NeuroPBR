import sys
import argparse
import torch
import coremltools as ct
from pathlib import Path

# Add parent directory to path to allow imports from training root
sys.path.append(str(Path(__file__).parent.parent))

from train import MultiViewPBRGenerator
from student.train import StudentGenerator

class MultiViewWrapper(torch.nn.Module):
    """
    Wraps the model to accept 3 separate image inputs (CoreML friendly)
    instead of a single 5D tensor (not supported by CoreML).
    Returns a tuple of outputs to ensure deterministic ordering for CoreML.
    """
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, view1, view2, view3):
        # Stack inputs: (1, 3, H, W) -> (1, 3, 3, H, W)
        x = torch.stack([view1, view2, view3], dim=1)
        out = self.model(x)
        
        # Normal map is [-1, 1]. We map it to [0, 1] first.
        normal_01 = (out['normal'] * 0.5) + 0.5
        
        # Core ML ImageType output expects [0, 255].
        # Since our model outputs [0, 1], we must scale up.
        return (
            out['albedo'] * 255.0,
            out['roughness'] * 255.0,
            out['metallic'] * 255.0,
            normal_01 * 255.0
        )

def load_model_from_checkpoint(checkpoint_path):
    """Load the correct model architecture and weights from a checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'config' key.")
    
    config = checkpoint["config"]
    state_dict = checkpoint.get("student_state_dict", checkpoint.get("generator_state_dict", checkpoint.get("state_dict")))
    
    if state_dict is None:
        raise ValueError("Could not find state_dict in checkpoint.")

    # Determine model class
    if config.model.encoder_type == "mobilenetv3":
        print(f"Detected Student Model (MobileNetV3)")
        model = StudentGenerator(config)
    else:
        print(f"Detected Standard Model ({config.model.encoder_type})")
        model = MultiViewPBRGenerator(config)

    # Clean up state dict (remove torch.compile prefixes)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    # Load weights
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    return model, config

def convert_to_coreml(
    model,
    config,
    output_path="pbr_model.mlpackage",
    skip_model_load=False,
    use_const_elimination=False,
    use_fp16=True,
):
    """
    Convert PyTorch model to CoreML format.

    Args:
       model: PyTorch model to convert
       output_path: Where to save the .mlpackage
       skip_model_load: Set True if running on incompatible OS
       use_const_elimination: Set True if model size is too large
       use_fp16: Set False if having artifacts
    """

    print("Tracing model...")
    wrapped_model = MultiViewWrapper(model.eval())
    
    # Create dummy input for tracing
    H, W = config.data.image_size
    
    # Force garbage collection before tracing
    import gc
    gc.collect()
    
    # Use 'with torch.no_grad()' to avoid storing gradients during trace
    with torch.no_grad():
        # Determine device for tracing
        # We use FP32 for tracing to avoid Core ML conversion issues with mixed precision ops (e.g. Hardswish).
        # Core ML will handle the FP16 quantization via compute_precision=ct.precision.FLOAT16.
        trace_device = "cpu"
        if torch.backends.mps.is_available():
             try:
                 print("Apple Silicon (MPS) detected. Tracing in FP32...")
                 trace_device = "mps"
                 wrapped_model = wrapped_model.to(trace_device).float()
             except Exception as e:
                 print(f"Failed to move to MPS: {e}. Falling back to CPU.")
                 trace_device = "cpu"
                 wrapped_model = wrapped_model.cpu().float()
        else:
             print(f"Tracing model on CPU (FP32)...")
             wrapped_model = wrapped_model.cpu().float()

        example_view = torch.rand(1, 3, H, W).to(trace_device)
        
        traced_model = torch.jit.trace(wrapped_model, (example_view, example_view, example_view))
        
        # Free memory immediately
        del example_view
        # Move model back to CPU to free MPS memory if needed, though not strictly necessary for script exit
        if trace_device == "mps":
            traced_model = traced_model.cpu()
        gc.collect()

    # Get normalization values
    if config.transform.use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = config.transform.mean
        std = config.transform.std

    # CoreML Scale/Bias calculation
    # CoreML: pixel = (input * scale) + bias
    # PyTorch: pixel = (input - mean) / std  =>  pixel = input * (1/std) - (mean/std)
    # scale = 1 / (255 * std)
    # bias = -mean / std
    
    avg_std = sum(std) / len(std)
    coreml_scale = 1.0 / (255.0 * avg_std)
    coreml_bias = [-m / s for m, s in zip(mean, std)]

    print(f"CoreML Normalization - Scale: {coreml_scale}, Bias: {coreml_bias}")

    # Define Inputs
    inputs = [
        ct.ImageType(name="view1", shape=(1, 3, H, W), scale=coreml_scale, bias=coreml_bias, color_layout=ct.colorlayout.RGB),
        ct.ImageType(name="view2", shape=(1, 3, H, W), scale=coreml_scale, bias=coreml_bias, color_layout=ct.colorlayout.RGB),
        ct.ImageType(name="view3", shape=(1, 3, H, W), scale=coreml_scale, bias=coreml_bias, color_layout=ct.colorlayout.RGB)
    ]

    # Define Outputs
    outputs = [
        ct.ImageType(name="albedo", color_layout=ct.colorlayout.RGB),
        ct.ImageType(name="roughness", color_layout=ct.colorlayout.GRAYSCALE),
        ct.ImageType(name="metallic", color_layout=ct.colorlayout.GRAYSCALE),
        ct.ImageType(name="normal", color_layout=ct.colorlayout.RGB)
    ]

    # Build optimized pass pipeline for memory efficiency
    # This enables operation fusion which reduces intermediate buffers
    pass_pipeline = ct.PassPipeline.DEFAULT
    
    # Add additional optimization passes for memory efficiency
    pass_pipeline.insert_pass(0, "common::add_int16_cast")  # Use smaller int types where possible
    
    if use_const_elimination:
        pass_pipeline.set_options("common::const_elimination", {"skip_const_by_size": "1e6"})

    precision = ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32

    print(f"Converting to CoreML (Precision: {precision})...")
    print("Enabling operation fusion for memory optimization...")
    
    mlmodel = ct.convert(
        model=traced_model,
        source='pytorch',
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
        compute_precision=precision,
        skip_model_load=skip_model_load,
        pass_pipeline=pass_pipeline,
    )

    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)
    
    # Verify size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    # For mlpackage, it's a directory. We need to sum up files.
    if Path(output_path).is_dir():
        size_mb = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file()) / (1024 * 1024)
        
    print(f"Model saved. Size: {size_mb:.2f} MB")
    if use_fp16 and size_mb > 15:
        print("WARNING: Model size seems large (>15MB) for a MobileNetV3 Student model.")
        print("It might still be in FP32. Check 'Compute Precision' in Xcode.")
    elif use_fp16:
        print("Size looks correct for FP16 (expecting ~10-12 MB).")
    
    return mlmodel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NeuroPBR checkpoint to CoreML")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="pbr_model.mlpackage", help="Output path")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 quantization")
    
    args = parser.parse_args()
    
    model, config = load_model_from_checkpoint(args.checkpoint)
    
    convert_to_coreml(
        model=model,
        config=config,
        output_path=args.output,
        use_fp16=not args.no_fp16
    )
    print("Done!")