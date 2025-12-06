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
    def __init__(self, original_model, mean, std):
        super().__init__()
        self.model = original_model
        
        # Pre-calculate normalization parameters for "baking" into the model
        # Input is [0, 255]. Target is (x/255 - mean) / std
        # Equivalent to: x * (1/(255*std)) - (mean/std)
        
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1)
        
        self.register_buffer('scale', 1.0 / (255.0 * std_tensor))
        self.register_buffer('bias', -mean_tensor / std_tensor)

    def forward(self, view1, view2, view3):
        # Normalize inputs (which are 0-255 from CoreML)
        v1 = view1 * self.scale + self.bias
        v2 = view2 * self.scale + self.bias
        v3 = view3 * self.scale + self.bias

        # Stack inputs: (1, 3, H, W) -> (1, 3, 3, H, W)
        x = torch.stack([v1, v2, v3], dim=1)
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
    
    # NOTE: We keep the Super-Resolution head intact.
    # The SR head was trained to upscale from 512 to 1024 with learned high-frequency details.
    # This produces better quality than generic Lanczos upscaling.
    # Output resolution will be 1024x1024 (upscaled from 512 internal resolution).
    
    model.eval()
    
    return model, config

def convert_to_coreml(
    model,
    config,
    output_path="pbr_model.mlpackage",
    skip_model_load=False,
    use_fp16=True,
    use_palettization=False,
):
    """
    Convert PyTorch model to CoreML format.

    Args:
       model: PyTorch model to convert
       output_path: Where to save the .mlpackage
       skip_model_load: Set True if running on incompatible OS
       use_fp16: Set False if having artifacts
       use_palettization: Set True to apply 8-bit weight palettization (reduces size but may affect quality)
    """

    # Get normalization values
    if config.transform.use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = config.transform.mean
        std = config.transform.std

    print("Tracing model...")
    # Pass normalization params to wrapper to bake them into the model graph
    wrapped_model = MultiViewWrapper(model.eval(), mean, std)
    
    # OPTIMIZATION: Use 512x512 input size for mobile
    # This is the maximum size that fits in iPhone memory.
    # The model outputs at 512x512, upscaled to 2048x2048 on-device.
    H, W = 512, 512
    
    # Modify encoder's first conv to use stride=1 instead of stride=2
    # This maintains reasonable internal resolution for 512x512 input
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'features'):
        first_conv = model.encoder.features[0][0]
        if hasattr(first_conv, 'stride') and first_conv.stride == (2, 2):
            print("OPTIMIZATION: Changing encoder first conv stride from 2 to 1 for 512x512 input.")
            import torch.nn as nn
            new_conv = nn.Conv2d(
                first_conv.in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=1,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            new_conv.weight.data = first_conv.weight.data.clone()
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()
            model.encoder.features[0][0] = new_conv
    
    # Force garbage collection before tracing
    import gc
    gc.collect()
    
    # Use 'with torch.no_grad()' to avoid storing gradients during trace
    with torch.no_grad():
        # Determine device for tracing
        trace_device = "cpu"
        
        # Try to trace in FP16 on MPS if requested
        if torch.backends.mps.is_available():
             try:
                 if use_fp16:
                     print("Apple Silicon (MPS) detected. Tracing in FP16 for ANE compatibility...")
                     trace_device = "mps"
                     wrapped_model = wrapped_model.to(trace_device).half()
                 else:
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

        # Create example input with correct dtype
        dtype = torch.float16 if (trace_device == "mps" and use_fp16) else torch.float32
        example_view = torch.rand(1, 3, H, W).to(trace_device).to(dtype)
        
        traced_model = torch.jit.trace(wrapped_model, (example_view, example_view, example_view))
        
        # Free memory immediately
        del example_view
        # Move model back to CPU to free MPS memory if needed, though not strictly necessary for script exit
        if trace_device == "mps":
            # IMPORTANT: When moving back to CPU, we must cast back to float32
            # CoreML conversion on CPU expects float32 inputs/weights even if it quantizes later
            # The error "bias has dtype fp16 whereas x has dtype fp32" happens because
            # coremltools sees the graph on CPU (where inputs are usually fp32) but weights are still fp16
            traced_model = traced_model.cpu().float()
        gc.collect()

    # Define Inputs
    # Note: scale and bias are now baked into the model, so we use default (scale=1.0, bias=0.0)
    # CoreML will pass [0, 255] values directly to the model inputs
    inputs = [
        ct.ImageType(name="view1", shape=(1, 3, H, W), color_layout=ct.colorlayout.RGB),
        ct.ImageType(name="view2", shape=(1, 3, H, W), color_layout=ct.colorlayout.RGB),
        ct.ImageType(name="view3", shape=(1, 3, H, W), color_layout=ct.colorlayout.RGB)
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
    
    # Constant elimination folds and removes redundant constant operations
    # This reduces model size and improves inference speed by pre-computing
    # constant expressions at compile time rather than runtime
    print("Enabling constant elimination optimization...")
    # Skip elimination for very large constants (>1MB) to avoid OOM during conversion
    pass_pipeline.set_options("common::const_elimination", {"skip_const_by_size": "1e6"})
    # Also enable dead code elimination to clean up unused operations
    pass_pipeline.insert_pass(1, "common::dead_code_elimination")

    precision = ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32

    print(f"Converting to CoreML (Precision: {precision})...")
    print("Enabling operation fusion for memory optimization...")
    
    mlmodel = ct.convert(
        model=traced_model,
        source='pytorch',
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=precision,
        skip_model_load=skip_model_load,
        pass_pipeline=pass_pipeline,
    )

    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)
    
    # Verify size before compression
    if Path(output_path).is_dir():
        size_mb = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file()) / (1024 * 1024)
    else:
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Model saved (before palettization). Size: {size_mb:.2f} MB")
    
    # Apply 8-bit palettization to further reduce model size and memory usage
    # This clusters weights into 256 unique values per tensor, reducing size by ~50%
    if use_palettization:
        print("Applying 8-bit weight palettization...")
        try:
            import coremltools.optimize as cto
            
            # Configure 8-bit palettization using k-means clustering
            op_config = cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=8)
            config = cto.coreml.OptimizationConfig(global_config=op_config)
            
            # Apply palettization
            mlmodel = cto.coreml.palettize_weights(mlmodel, config)
            
            # Save the compressed model
            mlmodel.save(output_path)
            
            # Verify compressed size
            if Path(output_path).is_dir():
                compressed_size_mb = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file()) / (1024 * 1024)
            else:
                compressed_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            
            print(f"Palettized model saved. Size: {compressed_size_mb:.2f} MB (was {size_mb:.2f} MB)")
            size_mb = compressed_size_mb
        except Exception as e:
            print(f"WARNING: Palettization failed: {e}")
            print("Continuing with uncompressed model.")
    
    if use_fp16 and size_mb > 15:
        print("WARNING: Model size seems large (>15MB) for a MobileNetV3 Student model.")
        print("It might still be in FP32. Check 'Compute Precision' in Xcode.")
    elif use_fp16:
        print("Size looks correct for FP16.")
    
    return mlmodel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NeuroPBR checkpoint to CoreML")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="pbr_model.mlpackage", help="Output path")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 quantization")
    parser.add_argument("--palettization", action="store_true", help="Enable 8-bit weight palettization (smaller model, may reduce quality)")
    
    args = parser.parse_args()
    
    model, config = load_model_from_checkpoint(args.checkpoint)
    
    convert_to_coreml(
        model=model,
        config=config,
        output_path=args.output,
        use_fp16=not args.no_fp16,
        use_palettization=args.palettization
    )
    print("Done!")