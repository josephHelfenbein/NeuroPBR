import coremltools as ct
import torch
from pathlib import Path

# RUN THIS FILE ON MACOS
# IF MLPROGRAM: macOS 12+ or iOS 15+
# IF neuralnetwork: macOS 10.13+ or iOS 11+
# For other systems, set skip_model_load=True to avoid compilation

''' USAGE:
# 1. Instantiate the standard class (Do NOT call torch.compile here)
model = MultiViewPBRGenerator(config) 

# 2. Load the weights
checkpoint = torch.load("path/to/your/checkpoint.pth", map_location="cpu")

# Extract state_dict (depends on how you saved it, sometimes it's under 'state_dict' key)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# --- CRITICAL FIX FOR COMPILED MODELS ---
# If you trained with torch.compile, the keys might look like "_orig_mod.encoder.conv1..."
# We need to remove "_orig_mod." so the raw model recognizes them.
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("_orig_mod.", "")  # Strip the compile prefix
    new_state_dict[new_key] = value

# 3. Load weights into the raw model
model.load_state_dict(new_state_dict)
model.eval()

# 4. NOW run the conversion function
mlmodel = convert_to_coreml(
        model=model,
        config=config,
        output_path="pbr_model.mlpackage",
        skip_model_load=False,
        use_const_elimination=False  # Set True if model >200MB
    )
'''

# This is because we can't use 5d tensors in coreml
class MultiViewWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, view1, view2, view3):
        # Stack inputs back into (1, 3, 3, H, W) for the internal model
        x = torch.stack([view1, view2, view3], dim=1)
        return self.model(x)

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

    wrapped_model = MultiViewWrapper(model.eval())

    # Trace the model
    example_view = torch.rand(1, 3, 2048, 2048)
    traced_model = torch.jit.trace(wrapped_model, (example_view, example_view, example_view))

    # Get normalization values from config
    if config.transform.use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = config.transform.mean
        std = config.transform.std

    coreml_bias = [-m / s for m, s in zip(mean, std)]

    avg_std = sum(std) / len(std)
    coreml_scale = 1.0 / (255.0 * avg_std)


    #--------INPUTS--------
    image_input1 = ct.ImageType(
        name="view1",
        shape=(1, 3, 2048, 2048),
        scale=coreml_scale,
        bias=coreml_bias,
        color_layout=ct.colorlayout.RGB,
        channel_first=True
    )

    image_input2 = ct.ImageType(
        name="view2",
        shape=(1, 3, 2048, 2048),
        scale=coreml_scale,
        bias=coreml_bias,
        color_layout=ct.colorlayout.RGB,
        channel_first=True
    )

    image_input3 = ct.ImageType(
        name="view3",
        shape=(1, 3, 2048, 2048),
        scale=coreml_scale,
        bias=coreml_bias,
        color_layout=ct.colorlayout.RGB,
        channel_first=True
    )


    #--------OUTPUTS--------
    image_albedo = ct.ImageType(
        name="albedo",
        shape=(1, 3, 2048, 2048),
        color_layout=ct.colorlayout.RGB,
        channel_first=True
    )

    image_roughness = ct.ImageType(
        name="roughness",
        shape=(1, 1, 2048, 2048),
        color_layout=ct.colorlayout.GRAYSCALE,
        channel_first=True
    )

    image_metallic = ct.ImageType(
        name="metallic",
        shape=(1, 1, 2048, 2048),
        color_layout=ct.colorlayout.GRAYSCALE,
        channel_first=True
    )

    image_normal = ct.ImageType(
        name="normal",
        shape=(1, 3, 2048, 2048),
        color_layout=ct.colorlayout.RGB,
        channel_first=True
    )

    #--------conversion--------
    # Set up pass pipeline if needed
    pipeline_option = ct.PassPipeline.CLEANUP
    if use_const_elimination:
        pipeline_option = ct.PassPipeline()
        pipeline_option.set_options("common::const_elimination", {"skip_const_by_size": "1e6"})

    precision = ct.precision.FLOAT32
    if use_fp16:
        precision = ct.precision.FLOAT16

    mlmodel = ct.convert(
        model=traced_model,
        source='pytorch',
        inputs=[image_input1, image_input2, image_input3],
        outputs=[image_albedo, image_roughness, image_metallic, image_normal],
        minimum_deployment_target= ct.target.iOS15,
        convert_to="mlprogram",
        compute_precision=precision,
        skip_model_load=skip_model_load,
        pass_pipeline=pipeline_option,
    )

    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    if Path(output_path).exists():
        size_mb = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file()) / (1024 ** 2)
        print(f"Model Size: {size_mb:.2f} MB")

    return mlmodel