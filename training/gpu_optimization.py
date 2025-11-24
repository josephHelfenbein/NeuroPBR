import torch
import os
import math
import warnings
import time

def detect_gpu_capabilities():
    """
    Figures out what kind of GPU we're working with.
    Returns a dictionary with the GPU name, memory, and compute capability.
    """
    if not torch.cuda.is_available():
        return {
            "is_available": False,
            "name": "CPU",
            "capability": (0, 0),
            "total_memory_gb": 0,
            "count": 0
        }

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    
    return {
        "is_available": True,
        "name": props.name,
        "capability": (props.major, props.minor),
        "total_memory_gb": props.total_memory / (1024**3),
        "count": torch.cuda.device_count()
    }

def apply_global_optimizations():
    """
    Sets up global PyTorch settings to make things run faster based on your GPU.
    """
    # Set allocator config to reduce fragmentation (must be done before CUDA init if possible)
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
         print("[GPU Optimization] Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation")
         os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    gpu_info = detect_gpu_capabilities()
    if not gpu_info["is_available"]:
        return

    major, minor = gpu_info["capability"]
    gpu_name = gpu_info["name"]

    print(f"[GPU Optimization] Detected: {gpu_name} (Compute Capability {major}.{minor})")

    # 1. Speed up math operations with TF32 (TensorFloat-32)
    # If we're on an Ampere GPU (RTX 30xx/A100) or newer (Compute Capability 8.0+),
    # we can use TF32. It makes matrix math way faster without losing much precision.
    if major >= 8:
        print("[GPU Optimization] Enabling TF32 for faster matrix multiplications (Ampere+)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # For newer PyTorch versions, this explicitly tells it to trade a tiny bit of precision for speed.
        if hasattr(torch, 'set_float32_matmul_precision'):
             torch.set_float32_matmul_precision('high')

    # 2. Enable cuDNN Benchmark Mode
    # This tells PyTorch to spend a little time at the start to find the absolute fastest 
    # algorithm for your specific convolution layer sizes. Great if input sizes don't change!
    # However, it can use extra memory. Disable if VRAM is tight.
    gpu_info = detect_gpu_capabilities()
    if gpu_info["is_available"] and gpu_info["total_memory_gb"] < 16:
         print("[GPU Optimization] Low VRAM detected: Disabling cuDNN benchmark mode to save memory")
         torch.backends.cudnn.benchmark = False
    else:
         print("[GPU Optimization] Enabling cuDNN benchmark mode")
         torch.backends.cudnn.benchmark = True

def get_amp_config():
    """
    Decides the best Mixed Precision (AMP) settings for your hardware.
    """
    gpu_info = detect_gpu_capabilities()
    if not gpu_info["is_available"]:
        return {"enabled": False, "dtype": torch.float32}

    major, minor = gpu_info["capability"]
    
    # If we have an Ampere GPU (RTX 30xx/A100) or newer, we should use BFloat16.
    # It's more stable than standard Float16 because it has the same dynamic range as Float32.
    if major >= 8:
        print("[GPU Optimization] Using BFloat16 for AMP (Ampere+)")
        return {"enabled": True, "dtype": torch.bfloat16}
    else:
        # For older GPUs (Volta/Turing/Pascal), we stick to standard Float16.
        print("[GPU Optimization] Using Float16 for AMP (Volta/Turing)")
        return {"enabled": True, "dtype": torch.float16}

def get_optimizer_params(device_type="cuda"):
    """
    Checks if we can use a 'fused' optimizer kernel.
    Fused optimizers combine multiple steps into one GPU kernel, which is faster.
    """
    if device_type != "cuda" or not torch.cuda.is_available():
        return {}

    gpu_info = detect_gpu_capabilities()
    major, _ = gpu_info["capability"]

    # Check if the AdamW optimizer supports the 'fused' argument.
    # This is usually available in newer PyTorch versions and works great on CUDA.
    import inspect
    if 'fused' in inspect.signature(torch.optim.AdamW).parameters:
        print("[GPU Optimization] Using fused optimizer kernel")
        return {"fused": True}
    
    return {}

def optimize_model_memory_format(model, device):
    """
    Switches the model's memory layout to 'channels_last'.
    NVIDIA Tensor Cores love this format (NHWC) and run significantly faster with it.
    """
    if device.type == "cuda":
        print("[GPU Optimization] Converting model to channels_last memory format")
        model = model.to(memory_format=torch.channels_last)
    return model

def calculate_optimal_batch_size(model_config, input_shape=(3, 2048, 2048), safety_margin=0.8):
    """
    Tries to guess the best batch size that will fit in your VRAM.
    
    Args:
        model_config: The model configuration object.
        input_shape: Shape of a single input sample (C, H, W).
        safety_margin: How much VRAM to use (0.8 means use 80%, leave 20% buffer).
    """
    gpu_info = detect_gpu_capabilities()
    if not gpu_info["is_available"]:
        return 2 # Fallback for CPU

    total_vram_gb = gpu_info["total_memory_gb"]
    
    # Let's do some rough math to estimate memory usage.
    
    # 1. Static Memory: Weights & Buffers
    # ResNet50 ~ 100MB, ViT ~ 100-200MB, Decoder ~ 50MB -> ~0.5GB just to load the model.
    model_static_gb = 0.5 
    
    # 2. Dynamic Memory: Activations & Gradients
    # This is the big one. Training requires storing intermediate values for backprop.
    # Memory usage scales linearly with pixel count.
    
    # Baseline: 1 sample (3 views) @ 1024x1024 ~ 8 GB VRAM (with AMP enabled)
    base_pixels = 1024 * 1024
    current_pixels = input_shape[1] * input_shape[2]
    scale_factor = current_pixels / base_pixels
    
    # Estimate per-sample memory
    # 8.0 GB base * scale_factor
    # For 2048x2048 (scale=4), this would be 32GB!
    # However, 8GB base was a very conservative estimate for ResNet50+ViT+Decoder.
    # Let's refine:
    # A 2048x2048 float16 layer (AMP) takes 2048*2048*2 bytes = 8MB per channel.
    # ResNet50 has many layers but spatial dim reduces quickly.
    # High-res layers (early ones) dominate memory.
    # 3 views * (Input + Conv1 + Layer1) is heavy.
    
    # Adjusted base for 1024x1024: ~6 GB
    gb_per_sample = 6.0 * scale_factor
    
    # Adjust based on model size (simple heuristic)
    if hasattr(model_config, 'encoder_backbone'):
        if 'resnet101' in model_config.encoder_backbone:
            gb_per_sample *= 1.2 # Bigger model = more memory
        elif 'resnet18' in model_config.encoder_backbone:
            gb_per_sample *= 0.6 # Smaller model = less memory

    # Adjust for lower VRAM cards (e.g. 12GB or less) which are more prone to fragmentation
    if total_vram_gb < 16:
        print(f"[GPU Optimization] Low VRAM detected ({total_vram_gb:.1f}GB). Increasing memory safety margins.")
        gb_per_sample *= 1.5  # Increased from 1.3 to 1.5
        safety_margin = min(safety_margin, 0.6) # Reduced from 0.7 to 0.6
        
        # Check for huge images on small GPU
        if input_shape[1] > 1024 or input_shape[2] > 1024:
            print(f"[GPU Optimization] ⚠️  WARNING: Image size {input_shape[1]}x{input_shape[2]} is very large for {total_vram_gb:.1f}GB VRAM.")
            print(f"[GPU Optimization]    Consider reducing config.data.image_size to (1024, 1024) or (512, 512).")
            
    available_mem = total_vram_gb * safety_margin - model_static_gb
    
    if available_mem <= 0:
        return 1
        
    batch_size = int(available_mem / gb_per_sample)
    batch_size = max(1, batch_size)
    
    # Round down to a power of 2 or even number if possible (GPUs like powers of 2)
    if batch_size > 4:
        batch_size = (batch_size // 2) * 2
        
    print(f"[GPU Optimization] Estimated optimal batch size: {batch_size} (VRAM: {total_vram_gb:.1f}GB)")
    return batch_size

def get_optimal_num_workers(batch_size):
    """
    Recommends how many CPU workers to use for data loading.
    Too few = GPU waits for data. Too many = CPU thrashing.
    """
    cpu_count = os.cpu_count() or 4
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Rule of thumb: 2-4 workers per GPU is usually the sweet spot.
    workers_per_gpu = 4
    total_workers = workers_per_gpu * gpu_count
    
    # We don't need more workers than the batch size (that would be wasteful).
    # But for heavy preprocessing, we might want a few extra.
    
    recommended = min(total_workers, cpu_count)
    
    # Always try to have at least 2 workers to keep the pipeline moving.
    recommended = max(2, recommended)
    
    # Leave one CPU core free for the main process/OS so the system doesn't freeze.
    if recommended >= cpu_count and cpu_count > 1:
        recommended = cpu_count - 1
        
    print(f"[GPU Optimization] Recommended num_workers: {recommended}")
    return recommended

def print_verification_commands():
    """
    Prints some handy commands you can run to check if everything is working.
    """
    print("\n" + "="*40)
    print("GPU Optimization Verification")
    print("="*40)
    print("1. Monitor GPU usage (run in terminal):")
    print("   watch -n 1 nvidia-smi")
    print("2. Check Tensor Cores usage (advanced):")
    print("   # Requires nsys or similar profiling tools")
    print("3. Verify PyTorch Settings in Python:")
    print("   import torch")
    print("   print(torch.backends.cuda.matmul.allow_tf32)")
    print("   print(torch.backends.cudnn.benchmark)")
    print("="*40 + "\n")

def apply_torch_compile(model, model_name="model"):
    """
    Smartly decides whether to use `torch.compile` (PyTorch 2.0+) on your model.
    
    It considers:
    - Is your PyTorch version new enough? (>=2.0)
    - Is your GPU powerful enough? (Ampere+ preferred)
    - Is this a spot instance? (If so, compile faster!)
    - Did you manually enable/disable it?
    
    Returns:
        compiled_model: The model (either compiled or original)
        compile_info: A dictionary telling you what happened and why.
    """
    
    # 1. Check User Preference (Environment Variable)
    # You can force this with `export USE_TORCH_COMPILE=false` (or true)
    use_compile_env = os.getenv('USE_TORCH_COMPILE', 'auto').lower()
    if use_compile_env == 'false':
        return model, {
            "status": "disabled", 
            "mode": None, 
            "reason": "User disabled via USE_TORCH_COMPILE=false",
            "warning": None
        }
    
    # 2. Check PyTorch Version
    # We need at least PyTorch 2.0 to use compile.
    torch_version = torch.__version__.split('.')
    major_version = int(torch_version[0])
    if major_version < 2:
        return model, {
            "status": "uncompiled", 
            "mode": None, 
            "reason": f"PyTorch < 2.0 (found {torch.__version__})",
            "warning": None
        }
        
    if not hasattr(torch, "compile"):
        return model, {
            "status": "uncompiled",
            "mode": None,
            "reason": "torch.compile not found",
            "warning": None
        }

    # 3. Detect Hardware
    gpu_info = detect_gpu_capabilities()
    if not gpu_info["is_available"]:
        return model, {
            "status": "uncompiled", 
            "mode": None, 
            "reason": "No GPU detected",
            "warning": None
        }
        
    major, minor = gpu_info["capability"]
    gpu_name = gpu_info["name"]
    
    # 4. Determine Compatibility & Mode
    mode = "default"
    reason = "Standard compilation"
    warning = None
    
    # Check for specific GPU architectures
    is_ampere_plus = major >= 8
    is_hopper = major >= 9
    is_turing = major == 7 and minor == 5
    is_volta = major == 7 and minor == 0
    
    # Spot instance detection
    # If we're on a spot instance, we might get preempted. 
    # We don't want to spend 10 minutes compiling if we only live for an hour.
    is_spot_instance = os.getenv('IS_SPOT_INSTANCE', 'false').lower() == 'true'
    
    if use_compile_env == 'true':
        # User said "DO IT!", so we do it.
        mode = "default" 
        reason = "User forced compilation"
    else:
        # Auto-detection logic: Let's be smart about this.
        
        # V100/T4 (Volta/Turing) - These older cards can struggle with dynamic shapes or have high overhead.
        if is_volta or (is_turing and "T4" in gpu_name):
            # Discriminators in GANs often have complex logic. On older cards, compiling them might hurt more than help.
            if "discriminator" in model_name.lower():
                 return model, {
                    "status": "uncompiled",
                    "mode": None,
                    "reason": "Skipping discriminator compilation on older GPU (V100/T4) to avoid overhead",
                    "warning": None
                }
        
        # H100/A100/RTX30/40 (Ampere+) - These are great candidates for compilation!
        if is_ampere_plus:
            total_memory_gb = gpu_info["total_memory_gb"]

            if is_spot_instance:
                # Fast compile mode for short-lived instances
                mode = "reduce-overhead"
                reason = "Spot instance detected - minimizing compile time"
                warning = "Spot instance: ensure training > 1 hour to amortize compilation cost"
            else:
                # For long training runs on good hardware, 'max-autotune' squeezes out every drop of performance.
                # But it takes longer to compile. We'll use it for the Generator (the heavy lifter).
                # However, max-autotune can be memory hungry. We'll only use it if we have plenty of VRAM (>= 20GB).
                
                if "generator" in model_name.lower() and total_memory_gb >= 20:
                    mode = "max-autotune"
                    reason = "High-end GPU with >20GB VRAM detected, optimizing for max performance"
                elif "generator" in model_name.lower():
                    mode = "default"
                    reason = "Ampere+ GPU detected, but VRAM < 20GB. Using default compile mode to save memory."
                else:
                    mode = "default"
        else:
            # Older GPUs or unknown - stick to safe defaults
            mode = "default"

    # 5. Attempt Compilation
    try:
        print(f"[GPU Optimization] Compiling {model_name} with mode='{mode}'...")
        start_time = time.time()
        
        # This is where the magic happens!
        compiled_model = torch.compile(model, mode=mode)
        
        # Note: The actual compilation happens "Just In Time" (JIT) when you first run data through it.
        # So the first training step will be slow, but subsequent ones will be fast.
        
        return compiled_model, {
            "status": "compiled",
            "mode": mode,
            "reason": reason,
            "warning": warning,
            "compile_time_estimate": "JIT (on first run)",
            "memory_overhead": "Unknown (JIT)"
        }
        
    except Exception as e:
        warnings.warn(f"torch.compile failed for {model_name}: {e}. Using uncompiled model.")
        return model, {
            "status": "uncompiled", 
            "mode": None, 
            "reason": f"Compilation failed: {str(e)}",
            "warning": None
        }

def verify_compilation(model, model_name):
    """
    Checks if the model is actually wrapped by torch.compile.
    """
    if hasattr(model, "_orig_mod"):
        print(f"[GPU Optimization] Verified: {model_name} is a compiled model wrapper.")
        return True
    else:
        # It might be uncompiled or compilation failed silently/gracefully
        return False

def optimize_resolution_for_vram(config):
    """
    Automatically adjusts image resolution if VRAM is limited.
    """
    gpu_info = detect_gpu_capabilities()
    if not gpu_info["is_available"]:
        return config

    total_vram_gb = gpu_info["total_memory_gb"]
    
    # User requested threshold: 16GB
    if total_vram_gb <= 16.0:
        print(f"[GPU Optimization] Detected {total_vram_gb:.1f}GB VRAM (<= 16GB).")
        print("[GPU Optimization] Auto-resizing inputs and outputs to 1024x1024 to prevent OOM.")
        
        # Update config
        config.data.image_size = (1024, 1024)
        config.data.output_size = (1024, 1024)
        
        # Ensure decoder doesn't try to upsample if we want 1024 output
        # If encoder stride is 1, output is same as input.
        # If encoder stride is 2, output is half input.
        # We want output to be 1024.
        
        if config.model.encoder_stride == 1:
            config.model.decoder_sr_scale = 0
        elif config.model.encoder_stride == 2:
            config.model.decoder_sr_scale = 2
            
    return config
