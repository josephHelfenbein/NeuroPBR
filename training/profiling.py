"""
Fixed memory profiler using psutil to track actual process memory.
Tracks real memory usage including PyTorch tensors.
"""

import torch
import torchvision.models as models
import time
import gc
import psutil
import os


def get_process_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # RSS = Resident Set Size
    return mem


def test_model_memory(model_name: str, model_fn, resolution: int = 2048):
    """Test a model's memory usage using psutil."""

    print(f"\n{'='*60}")
    print(f"Testing: {model_name} @ {resolution}×{resolution}")
    print(f"{'='*60}")

    # Force garbage collection
    gc.collect()

    # Measure baseline memory
    baseline_memory = get_process_memory_mb()
    print(f"Baseline memory: {baseline_memory:.1f} MB")

    # Load model
    print("Loading model...")
    model = model_fn(weights=None)  # Fix deprecation warning
    model.eval()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    param_mb = params * 4 / (1024**2)  # FP32
    print(f"Parameters: {params:,} ({param_mb:.1f} MB)")

    after_model = get_process_memory_mb()
    model_memory = after_model - baseline_memory
    print(f"Memory after loading model: +{model_memory:.1f} MB")

    # Create input
    print(f"Creating {resolution}×{resolution} input...")
    input_tensor = torch.randn(1, 3, resolution, resolution)
    input_mb = input_tensor.numel() * 4 / (1024**2)
    print(f"Input size: {input_mb:.1f} MB")

    after_input = get_process_memory_mb()
    input_memory = after_input - after_model
    print(f"Memory after creating input: +{input_memory:.1f} MB")

    # Run inference and measure peak
    print("Running inference...")
    start_time = time.time()

    with torch.no_grad():
        output = model(input_tensor)

    elapsed = time.time() - start_time

    # Measure peak (right after inference)
    peak_memory = get_process_memory_mb()
    peak_delta = peak_memory - baseline_memory
    activation_memory = peak_memory - after_input

    print(f"\nResults:")
    print(f"  ├─ Baseline: {baseline_memory:.1f} MB")
    print(f"  ├─ Model weights: +{model_memory:.1f} MB")
    print(f"  ├─ Input tensor: +{input_memory:.1f} MB")
    print(f"  ├─ Activations (peak): +{activation_memory:.1f} MB")
    print(f"  ├─ TOTAL PEAK: {peak_memory:.1f} MB ({peak_delta:.1f} MB delta)")
    print(f"  ├─ Inference time: {elapsed:.2f}s")
    print(
        f"  └─ Fits in 1800 MB budget? {'YES' if peak_delta < 1800 else 'NO'}")

    # Cleanup
    del model, input_tensor, output
    gc.collect()

    return peak_delta


if __name__ == "__main__":
    print("NeuroPBR Memory Profiling on Mac (using psutil)")
    print("Testing different backbones...\n")

    tests = [
        ("MobileNetV2", models.mobilenet_v2),
        ("MobileNetV3-Large", models.mobilenet_v3_large),
        ("ResNet18", models.resnet18),
    ]

    results = {}

    # Test at different resolutions
    for resolution in [512, 1024, 2048]:
        print(f"\n{'#'*60}")
        print(f"# RESOLUTION: {resolution}×{resolution}")
        print(f"{'#'*60}")

        for name, model_fn in tests:
            try:
                peak = test_model_memory(name, model_fn, resolution)
                results[f"{name}_{resolution}"] = peak
            except Exception as e:
                print(f"Error: {e}")

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY - Peak Memory Usage (Delta from Baseline)")
    print(f"{'='*60}\n")

    for key, value in results.items():
        name, res = key.rsplit('_', 1)
        status = "YES" if value < 1800 else "NO"
        print(f"{status} {name:20s} @ {res:4s}×{res:4s}: {value:6.1f} MB")

    print(f"\nNote: These are CPU measurements on Mac.")
    print(f"   iPhone Neural Engine may use different amounts.")
    print(f"   Consider these as upper bounds (FP32). FP16 would be ~50% less.")