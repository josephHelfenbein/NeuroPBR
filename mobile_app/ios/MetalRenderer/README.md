# MetalRenderer

C++ and Metal-based renderer for real-time visualization inside the iOS app.

## Architecture snapshot

- **C++ Core (`Renderer.hpp/.cpp`)** – scene graph, material slots, camera, and toggle/state management. Emits compact uniform blocks that stay platform agnostic and exposes a flat C API the mobile layers can call from Objective-C++.
- **Metal shaders (`MetalShaders.metal`)** – PBR shading with GGX, Smith geometry, Schlick Fresnel, split-sum specular IBL, irradiance. Supports multiple geometry types (Sphere, Cube, Plane) and tone mapping operators (ACES, Filmic).
- **Compute prefilter (`EnvironmentPrefilter.metal/.mm`)** – Metal compute kernels that convert equirectangular HDRIs into cubemaps and precompute irradiance, specular prefilter mip chains, and the BRDF LUT so mobile builds no longer need the CUDA preprocessing step.
- **Objective-C++ bridge (`MetalBridge.mm`)** – owns Metal objects (device, queue, heaps, textures), translates Core ML outputs into `MTLTexture` updates, caches prefiltered HDRIs, and publishes rendered textures through callbacks to Flutter.
- **Flutter plugin (`neuropbr_plugin.dart`)** – wraps a `MethodChannel` to control renderer lifecycle, environment selection, toggles, and to bind the native texture into Flutter’s `Texture` widget.

### Data flow overview

1. Flutter requests an init with target size → the Objective-C++ bridge provisions Metal resources and the C++ renderer state.
2. Core ML outputs (NSData, files, or GPU textures) are pushed through `updateMaterial`, where the bridge reuses/updates `MTLTexture` storage and notifies the C++ core so its descriptor hashes stay in sync.
3. Every UI tick `renderFrame` builds uniform buffers from the C++ core, encodes the lighting pass, and presents the drawable that Flutter’s texture registry displays. If only an HDR equirect is provided, the bridge runs the compute prefilter once to populate the irradiance/specular/BRDF caches before render.
4. Optional readback exports RGBA frames (PNG/JPEG) via an Objective-C++ callback, making it easy to share previews or download results.

Additional details – build steps, Flutter integration, and CLI/automation snippets – live further down once implementation lands.

## Integration (Reference)

> **Note:** The `mobile_app` in this repository is already configured using a local CocoaPod. These steps are provided for reference if you need to integrate the renderer into a new project manually.

1. **Wire files into the Xcode target**
	- Add `Renderer.hpp/.cpp`, `MetalBridge.mm`, `NeuropbrMetalRendererPlugin.h`, `EnvironmentPrefilter.h/.mm`, `EnvironmentPrefilter.metal`, and `MetalShaders.metal` to the `Runner` (or host app) target.
	- Ensure `MetalShaders.metal` and `EnvironmentPrefilter.metal` are compiled into the default Metal library.
	- Enable the *Metal API Validation* option in the scheme only for debug builds to keep release builds lean.

2. **Objective-C++ bridge registration**
	- `MetalBridge.mm` already implements `NeuropbrMetalRendererPlugin`. Flutter’s generated registrant will automatically invoke it when the file is part of the iOS target.
	- No additional Swift/ObjC glue is required unless the project opts out of the generated registrant; in that case call `[NeuropbrMetalRendererPlugin registerWithRegistrar:self]` from `AppDelegate`.

3. **Flutter side**
	- Import `neuropbr_plugin.dart` and obtain the singleton `NeuropbrRenderer.instance`.
	- Display the Metal output through `NeuropbrRenderer.instance.buildPreviewTexture()` inside the Flutter tree.
	- Use `NeuropbrMaterialTextures` and `NeuropbrEnvironment` helpers to wrap Core ML outputs or on-disk assets before passing them over the `MethodChannel`.

4. **Required entitlements / Info.plist keys**
	- If you call `exportFrame()` and write to the camera roll, add `NSPhotoLibraryAddUsageDescription` to `ios/Runner/Info.plist` with a short justification string.
	- HDRI files shipped in-app should be listed under “Copy Bundle Resources”; no extra entitlements are necessary for Metal or Core ML usage beyond enabling the “Metal” capability in your provisioning profile.

## Running the demo helpers

```bash
cd mobile_app
flutter pub get
flutter run
```

- The Dart helper `lib/neuropbr_preview_demo.dart` demonstrates init → upload textures → render → export. Invoke `runNeuropbrPreviewDemo()` from any Flutter `main()` or integration test to smoke-test the flow without UI wiring.
- When Core ML pumps out fresh maps, call `NeuropbrRenderer.updateMaterialTexture` with the slot you wish to update. The bridge keeps the underlying `MTLTexture` allocations alive and reuses `replaceRegion` uploads for low latency.

## MethodChannel surface

| Method                | Arguments (Map)                                                                | Notes |
|-----------------------|---------------------------------------------------------------------------------|-------|
| `initRenderer`        | `width`, `height`                                                             | Returns `{textureId, device}`. |
| `loadMaterial`        | `materialId`, `textures{albedo, normal, roughness, metallic}`     | Accepts either raw bytes or absolute file paths per slot. |
| `updateMaterial`      | `materialId`, `slot`, `payload`                                               | Updates a single texture in-place. |
| `setCamera`           | `position`, `target`, `up`, `fov`, `near`, `far`                              | Defaults to look-at origin if omitted. |
| `setLighting`         | `exposure`, `intensity`, `rotation`                                           | Rotation turns the env map around +Y. |
| `setPreview`          | Tint, multipliers, toggles, channel, `toneMapping`, `zoom`, `modelType`       | Matches `NeuropbrPreviewControls`. Tone mapping: 0=ACES, 1=Filmic. |
| `setModelType`        | `type` (0=Sphere, 1=Cube, 2=Plane)                                            | Switches the geometry used for preview. |
| `setEnvironment`      | `environmentId`, file paths for env/irradiance/prefilter/brdf **or** `hdr` + optional sampling overrides | Provide cubemaps directly, or pass `hdr` (path/payload) to trigger on-device prefiltering. |
| `renderFrame`         | `materialId`                                                                  | Emits `onFrameRendered` via the same channel when finished. |
| `exportFrame`         | `format` (currently `png`)                                                    | Returns raw PNG bytes. |

Native callbacks use the same channel (`neuropbr_renderer`) with method names `onFrameRendered` and `onRendererError`. The Dart plugin already forwards those into Stream controllers.

## Minimal usage script

See `mobile_app/lib/neuropbr_preview_demo.dart` for an executable walkthrough. The sequence is:

1. `NeuropbrRenderer.instance.initRenderer(width, height)`
2. `setCamera`, `setLighting`, `setPreviewControls`
3. `loadMaterial` with Core ML outputs wrapped in `NeuropbrMaterialTextures`
4. `setEnvironment` to point at the HDRI/prefilter LUT bundle assets
5. `renderFrame` and optionally `exportFrame` to obtain PNG bytes (or watch the on-frame stream to drive your Flutter UI)

Drop the resulting `Texture` widget anywhere in the Flutter tree to embed the live Metal surface inside the app UI.

## Precomputed Environment Maps

For faster app startup, the mobile app now uses **precomputed environment maps** instead of processing raw HDRIs at runtime.

### How It Works

The app loads precomputed KTX files from `assets/env_maps/` containing:
- `{name}_env.ktx` – Environment cubemap (512×512 per face, RGBA16F)
- `{name}_irradiance.ktx` – Diffuse irradiance cubemap (32×32 per face)  
- `{name}_prefiltered.ktx` – Specular prefiltered cubemap with mip levels for roughness
- `brdf_lut.ktx` – Shared BRDF integration lookup table (512×512, RG16F)

The `EnvironmentPrefilter` class includes KTX loading methods:
- `loadKTXCubemap:error:` – Parses KTX cubemap files with mipmap support
- `loadKTX2DTexture:error:` – Parses KTX 2D textures (for BRDF LUT)
- `loadPrecomputedEnvironment:irradiancePath:prefilteredPath:brdfPath:error:` – Convenience method to load all environment textures at once

`MetalBridge.mm` automatically detects `.ktx` file extensions and routes them through the KTX loader instead of `MTKTextureLoader`.

### Regenerating Environment Maps

If you need to add new HDRIs or regenerate the environment maps:

1. Place `.hdr` files in `mobile_app/assets/hdris/`
2. Run the generation script (requires macOS with Metal):
   ```bash
   cd mobile_app/scripts
   ./generate_env_maps.sh
   ```

This script uses the same Metal compute shaders (`EnvironmentPrefilter.metal`) to precompute the maps on the GPU, processing all HDRIs in ~2 seconds.

### Live HDRI Processing (Fallback)

The renderer still supports on-device HDRI processing as a fallback:

- Supplying `hdr` (path or payload) inside `setEnvironment` tells the bridge to load the equirect texture, run `EnvironmentPrefilter.metal` compute kernels to produce cubemap, irradiance, prefiltered mip chain, and BRDF LUT.
- If you provide `environment` pointing to a `.hdr/.exr` file while other slots are omitted, the bridge automatically enables this pipeline.
- Optional knobs (`faceSize`, `irradianceSize`, `specularSamples`, `diffuseSamples`) can trade quality for speed.
- Generated textures are cached inside the bridge, so subsequent renders reuse the prefiltered results without recomputing.

This is useful for dynamically loading user-provided HDRIs, but adds ~1-2 seconds to initial environment setup compared to precomputed maps.
