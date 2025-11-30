# Renderer

C++/CUDA renderer for generating synthetic training data using image-based lighting (IBL).

- Loads materials from the `dataset/` folder.
- Renders 3 randomized HDRI-lit views per material.
- Outputs paired (input renders + ground-truth PBR maps) for model training.

## Implementation Details

The renderer is built using **C++17** and **CUDA**, implementing a standard PBR pipeline optimized for high-throughput data generation.

### Multithreaded Pipeline & GPU Batching
To maximize GPU utilization and minimize I/O bottlenecks, the renderer uses a 3-stage multithreaded pipeline connected by thread-safe queues. The depth of this pipeline (the "batch size") is dynamically calculated at runtime based on available resources.

1.  **Loader Thread:** Reads material textures (Albedo, Normal, Roughness, Metallic) from disk, packs them, and uploads them directly into pre-allocated GPU memory slots.
2.  **Render Thread:** Consumes requests, executes the CUDA rendering kernels on the pre-loaded data, and downloads the results.
3.  **Writer Thread:** Saves the rendered images to disk as PNGs and recycles the GPU memory slots for new requests.

### Dynamic Resource Management
The renderer automatically detects available **System RAM** and **GPU VRAM** to determine the optimal batch size:
-   **CPU Batch Limit:** Calculated to ensure enough RAM for loading textures and buffering output frames.
-   **GPU Batch Limit:** Calculated to ensure all in-flight materials fit within VRAM (approx. 224MB per slot for 2K textures).
-   **Pre-allocation:** The renderer pre-allocates all necessary GPU buffers at startup based on the determined batch size, eliminating runtime allocation overhead and fragmentation.

This ensures the renderer runs at maximum throughput on high-end systems while remaining stable on hardware with limited memory.

### Shading Model
It uses the **Cook-Torrance** microfacet specular shading model, which is the industry standard for PBR:
- **Distribution (D):** Trowbridge-Reitz (GGX)
- **Geometry (G):** Smith (Schlick-GGX)
- **Fresnel (F):** Schlick approximation

### Image-Based Lighting (IBL)
Lighting is purely image-based, using the **Split-Sum Approximation** to efficiently evaluate the lighting integral:
1.  **Irradiance Map:** A diffuse convolution of the environment map.
2.  **Prefiltered Environment Map:** Specular reflection pre-calculated at different roughness levels (stored in mipmaps).
3.  **BRDF Integration LUT:** A precomputed 2D texture storing the scale and bias for the Fresnel term.

### Data Augmentation
To make the neural network robust to real-world imperfections, the renderer generates two types of data:
- **Clean:** Perfect PBR rendering.
- **Dirty:** Adds randomized synthetic degradations:
    -   **Shadows:** Procedurally simulated occlusion to mimic uneven lighting.
    -   **Camera Artifacts:** Procedurally simulated lens smudges and scratches.

### CUDA Kernels

The rendering logic is distributed across several optimized CUDA kernels:

-   `shadeKernel`: The primary ray-casting kernel. It computes the camera ray for each pixel, intersects it with the material plane, and evaluates the PBR shading model. It also handles the procedural generation of shadows and camera artifacts on the fly.
-   `equirectangularToCubemap`: Converts input HDRI images (equirectangular projection) into cubemaps for efficient sampling.
-   `convolveDiffuseIrradiance`: Computes the diffuse irradiance map by convolving the environment map with a cosine-weighted hemisphere.
-   `prefilterSpecularCubemap`: Generates the pre-filtered environment map for specular reflections, using importance sampling (GGX) at varying roughness levels.
-   `computeEnvironmentBrightness`: Analyzes the HDRI to determine horizon and zenith brightness, which drives the procedural shadow intensity.
-   `precomputeBRDF`: Generates the 2D LUT for the Split-Sum approximation.

## Prerequisites

- CMake 3.18 or newer
- NVIDIA CUDA Toolkit (matching the GPU in your system)
- A C++17 compiler with CUDA support (MSVC, Clang, or GCC + NVCC)

If you cloned without submodules, run:

```bash
git submodule update --init --recursive
```

## Build & Run

### Linux / WSL2 (Recommended)

Ensure you have `cmake`, `build-essential` (GCC), and the NVIDIA CUDA Toolkit installed.

```bash
cd renderer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The binary will be written to `bin/neuropbr_renderer`.

### Command-line usage

```bash
./bin/neuropbr_renderer <materials_dir> <num_samples> [--start-index N]
```

- `<materials_dir>` – Path to the cleaned material dataset (each material folder must contain `albedo.png`, `normal.png`, `roughness.png`, `metallic.png` and be uniquely named).
- `<num_samples>` – Number of samples to render; each sample produces three views and writes to `output/clean` or `output/dirty` plus `output/render_metadata.json`.
- `--start-index` / `-s` – Optional offset applied to the generated sample folders (`sample_<index>`). Use this to append new renders to an existing dataset without overwriting earlier samples. Defaults to `0`.

Example (from `renderer/`):

```bash
./bin/neuropbr_renderer ../dataset/matsynth_clean 2000 --start-index 6000
```

Ensure `assets/hdris` contain the required textures before rendering.

### Visual Studio (Windows Alternative)

If you must build on Windows, use the Visual Studio generator:

```bat
cmake -G "Visual Studio 17 2022" -A x64 -T host=x64 -S . -B build
cmake --build build --config Release --parallel
```
The binary will be at `bin/Release/neuropbr_renderer.exe`.
