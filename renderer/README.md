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

### Visual Studio 2022 + NVCC (Windows)

From a **Developer Command Prompt for VS 2022** (or Developer PowerShell), run:

```bat
cd renderer
cmake -G "Visual Studio 17 2022" -A x64 -T host=x64 -S . -B build
cmake --build build --config Release --parallel
```

The binary will be written to `bin/Release/neuropbr_renderer.exe`. If you want a Debug build, replace the last line with `--config Debug`.

### Command-line usage

```bash
neuropbr_renderer.exe <materials_dir> <num_samples>
```

- `<materials_dir>` – Path to the cleaned material dataset (each folder must contain `albedo.png`, `normal.png`, `roughness.png`, `metallic.png`).
- `<num_samples>` – Number of samples to render; each sample produces three views and writes to `output/clean` or `output/dirty` plus `output/render_metadata.json`.

Example (from `renderer/`):

```powershell
.\bin\Release\neuropbr_renderer.exe ..\\dataset\\matsynth_clean 2000
```

Ensure `assets/hdris`, `assets/camerasmudges`, and `assets/lensflares` contain the required textures before rendering.

### Ninja / Make (optional)

If you have the CUDA toolkit configured with a GCC/Clang host compiler, you can use a single-config generator instead:

```bash
cd renderer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The executable in this case is produced as `bin/neuropbr_renderer`.
