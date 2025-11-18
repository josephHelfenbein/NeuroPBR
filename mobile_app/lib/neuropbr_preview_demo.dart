import 'dart:typed_data';

import 'neuropbr_plugin.dart';

/// Quick-start helper showing how to feed Core ML outputs into the renderer.
Future<void> runNeuropbrPreviewDemo() async {
  final renderer = NeuropbrRenderer.instance;
  final textureId = await renderer.initRenderer(width: 1024, height: 768);
  renderer.onFrameRendered.listen((_) => print('Frame available for texture $textureId'));
  renderer.onRendererError.listen((message) => print('Renderer error: $message'));

  await renderer.setCamera(const NeuropbrCamera(
    position: [0.0, 0.0, 4.0],
    target: [0.0, 0.0, 0.0],
  ));

  await renderer.setLighting(const NeuropbrLighting(exposure: 0.0, intensity: 1.2, rotation: 0.25));

  await renderer.setPreviewControls(const NeuropbrPreviewControls(
    tint: [1.0, 1.0, 1.0],
    roughnessMultiplier: 1.0,
    metallicMultiplier: 1.0,
  ));

  // Mock data, swap in Core ML output bytes/paths.
  final fakeFloat32 = Float32List.fromList(List<double>.filled(256 * 256 * 4, 0.5));
  final textureBytes = Uint8List.view(fakeFloat32.buffer);

  await renderer.loadMaterial('demo', NeuropbrMaterialTextures(
    albedo: NeuropbrTexturePayload.fromBytes(
      textureBytes,
      width: 256,
      height: 256,
      format: 'rgba32float',
    ),
    normal: NeuropbrTexturePayload.fromBytes(
      textureBytes,
      width: 256,
      height: 256,
      format: 'rgba32float',
    ),
    roughness: NeuropbrTexturePayload.fromBytes(
      textureBytes,
      width: 256,
      height: 256,
      format: 'r32float',
      channels: 1,
    ),
    metallic: NeuropbrTexturePayload.fromBytes(
      textureBytes,
      width: 256,
      height: 256,
      format: 'r32float',
      channels: 1,
    ),
  ));

  await renderer.setEnvironment(const NeuropbrEnvironment(
    environmentPath: 'Assets/cubemaps/studio.ktx2',
    irradiancePath: 'Assets/cubemaps/studio_irr.ktx2',
    prefilteredPath: 'Assets/cubemaps/studio_prefilter.ktx2',
    brdfPath: 'Assets/LUTs/brdf_lut.ktx2',
  ));

  await renderer.renderFrame('demo');

  final png = await renderer.exportFrame();
  if (png != null) {
    print('Captured preview (${png.length} bytes).');
  }
}
