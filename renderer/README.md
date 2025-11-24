# Renderer

C++/CUDA renderer for generating synthetic training data using image-based lighting (IBL).

- Loads materials from the `dataset/` folder.
- Renders 3 randomized HDRI-lit views per material.
- Outputs paired (input renders + ground-truth PBR maps) for model training.


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
./bin/neuropbr_renderer <materials_dir> <num_samples>
```

- `<materials_dir>` – Path to the cleaned material dataset (each folder must contain `albedo.png`, `normal.png`, `roughness.png`, `metallic.png`).
- `<num_samples>` – Number of samples to render; each sample produces three views and writes to `output/clean` or `output/dirty` plus `output/render_metadata.json`.

Example (from `renderer/`):

```bash
./bin/neuropbr_renderer ../dataset/matsynth_clean 2000
```

Ensure `assets/hdris` contain the required textures before rendering.

### Visual Studio (Windows Alternative)

If you must build on Windows, use the Visual Studio generator:

```bat
cmake -G "Visual Studio 17 2022" -A x64 -T host=x64 -S . -B build
cmake --build build --config Release --parallel
```
The binary will be at `bin/Release/neuropbr_renderer.exe`.
