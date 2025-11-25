import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

enum NeuropbrToneMapping { aces, filmic }
enum NeuropbrModelType { sphere, cube, plane }

/// High-level controller wrapping the native Metal renderer bridge.
class NeuropbrRenderer {
  NeuropbrRenderer._internal() {
    _channel.setMethodCallHandler(_handleCallback);
  }

  static final NeuropbrRenderer instance = NeuropbrRenderer._internal();

  final MethodChannel _channel = const MethodChannel('neuropbr_renderer');
  final StreamController<int> _frameStream = StreamController<int>.broadcast();
  final StreamController<String> _errorStream = StreamController<String>.broadcast();

  int? _textureId;
  bool _initialized = false;

  Stream<int> get onFrameRendered => _frameStream.stream;
  Stream<String> get onRendererError => _errorStream.stream;
  int? get textureId => _textureId;
  bool get isReady => _initialized && _textureId != null;

  Future<int> initRenderer({required int width, required int height}) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('initRenderer', {
      'width': width,
      'height': height,
    });
    _textureId = result?['textureId'] as int?;
    _initialized = _textureId != null;
    if (_textureId == null) {
      throw StateError('Renderer initialization failed: no texture id returned.');
    }
    return _textureId!;
  }

  Widget buildPreviewTexture({Key? key, FilterQuality filterQuality = FilterQuality.medium}) {
    final id = _textureId;
    if (id == null) {
      throw StateError('Renderer not initialized. Call initRenderer first.');
    }
    return Texture(key: key, textureId: id, filterQuality: filterQuality);
  }

  Future<void> setCamera(NeuropbrCamera camera) async {
    await _channel.invokeMethod<void>('setCamera', camera.toMap());
  }

  Future<void> setLighting(NeuropbrLighting lighting) async {
    await _channel.invokeMethod<void>('setLighting', lighting.toMap());
  }

  Future<void> setPreviewControls(NeuropbrPreviewControls preview) async {
    await _channel.invokeMethod<void>('setPreview', preview.toMap());
  }

  Future<void> loadMaterial(String materialId, NeuropbrMaterialTextures textures) async {
    await _channel.invokeMethod<void>('loadMaterial', {
      'materialId': materialId.hashCode,
      'textures': textures.toMap(),
    });
  }

  Future<void> updateMaterialTexture(String materialId, String slot, NeuropbrTexturePayload payload) async {
    await _channel.invokeMethod<void>('updateMaterial', {
      'materialId': materialId.hashCode,
      'slot': slot,
      'payload': payload.toMap(),
    });
  }

  Future<void> setEnvironment(NeuropbrEnvironment environment) async {
    await _channel.invokeMethod<void>('setEnvironment', environment.toMap());
  }

  Future<void> renderFrame(String materialId) async {
    await _channel.invokeMethod<void>('renderFrame', {
      'materialId': materialId.hashCode,
    });
  }

  Future<Uint8List?> exportFrame({String format = 'png'}) {
    return _channel.invokeMethod<Uint8List>('exportFrame', {'format': format});
  }

  Future<void> dispose() async {
    await _frameStream.close();
    await _errorStream.close();
  }

  Future<void> _handleCallback(MethodCall call) async {
    switch (call.method) {
      case 'onFrameRendered':
        final textureId = (call.arguments as Map?)?['textureId'] as int?;
        if (textureId != null) {
          _frameStream.add(textureId);
        }
        break;
      case 'onRendererError':
        final message = (call.arguments as Map?)?['message'] as String?;
        if (message != null) {
          _errorStream.add(message);
        }
        break;
      default:
        break;
    }
  }

  Future<void> setModelType(NeuropbrModelType type) async {
    await _channel.invokeMethod<void>('setModelType', {'type': type.index});
  }
}

class NeuropbrCamera {
  const NeuropbrCamera({
    this.position = const [0.0, 0.0, 5.0],
    this.target = const [0.0, 0.0, 0.0],
    this.up = const [0.0, 1.0, 0.0],
    this.fov = 45.0,
    this.near = 0.01,
    this.far = 100.0,
  });

  final List<double> position;
  final List<double> target;
  final List<double> up;
  final double fov;
  final double near;
  final double far;

  Map<String, dynamic> toMap() => {
        'position': position,
        'target': target,
        'up': up,
        'fov': fov,
        'near': near,
        'far': far,
      };
}

class NeuropbrLighting {
  const NeuropbrLighting({
    this.exposure = 0.0,
    this.intensity = 1.0,
    this.rotation = 0.0,
  });

  final double exposure;
  final double intensity;
  final double rotation;

  Map<String, dynamic> toMap() => {
        'exposure': exposure,
        'intensity': intensity,
        'rotation': rotation,
      };
}

class NeuropbrPreviewControls {
  const NeuropbrPreviewControls({
    this.tint = const [1.0, 1.0, 1.0],
    this.roughnessMultiplier = 1.0,
    this.metallicMultiplier = 1.0,
    this.enableNormal = true,
    this.showNormals = false,
    this.showWireframe = false,
    this.channel = 0,
    this.toneMapping = NeuropbrToneMapping.aces,
    this.modelType = NeuropbrModelType.sphere,
    this.zoom = 1.0,
  });

  final List<double> tint;
  final double roughnessMultiplier;
  final double metallicMultiplier;
  final bool enableNormal;
  final bool showNormals;
  final bool showWireframe;
  final int channel;
  final NeuropbrToneMapping toneMapping;
  final NeuropbrModelType modelType;
  final double zoom;

  Map<String, dynamic> toMap() => {
        'tint': tint,
        'roughnessMultiplier': roughnessMultiplier,
        'metallicMultiplier': metallicMultiplier,
        'enableNormal': enableNormal,
        'showNormals': showNormals,
        'showWireframe': showWireframe,
        'channel': channel,
        'toneMapping': toneMapping.index,
        'modelType': modelType.index,
        'zoom': zoom,
      };
}

class NeuropbrTexturePayload {
  const NeuropbrTexturePayload({
    this.bytes,
    this.path,
    required this.width,
    required this.height,
    this.format = 'rgba32float',
    this.channels,
    this.isCube = false,
  }) : assert(bytes != null || path != null, 'Provide either raw bytes or a file path.');

  NeuropbrTexturePayload.fromBytes(
    Uint8List data, {
    required int width,
    required int height,
    String format = 'rgba32float',
    int? channels,
  }) : this(bytes: data, width: width, height: height, format: format, channels: channels);

  NeuropbrTexturePayload.fromFile(
    String filePath, {
    required int width,
    required int height,
    String format = 'rgba16float',
    int? channels,
    bool isCube = false,
  }) : this(path: filePath, width: width, height: height, format: format, channels: channels, isCube: isCube);

  final Uint8List? bytes;
  final String? path;
  final int width;
  final int height;
  final String format;
  final int? channels;
  final bool isCube;

  Map<String, dynamic> toMap() => {
        if (bytes != null) 'bytes': bytes,
        if (path != null) 'path': path,
        'width': width,
        'height': height,
        'format': format,
        if (channels != null) 'channels': channels,
        if (isCube) 'isCube': true,
      };
}

class NeuropbrMaterialTextures {
  const NeuropbrMaterialTextures({
    this.albedo,
    this.normal,
    this.roughness,
    this.metallic,
  });

  final NeuropbrTexturePayload? albedo;
  final NeuropbrTexturePayload? normal;
  final NeuropbrTexturePayload? roughness;
  final NeuropbrTexturePayload? metallic;

  Map<String, dynamic> toMap() => {
        if (albedo != null) 'albedo': albedo!.toMap(),
        if (normal != null) 'normal': normal!.toMap(),
        if (roughness != null) 'roughness': roughness!.toMap(),
        if (metallic != null) 'metallic': metallic!.toMap(),
      };
}

class NeuropbrEnvironment {
  const NeuropbrEnvironment({
    this.environmentId = 0,
    this.environmentPath,
    this.irradiancePath,
    this.prefilteredPath,
    this.brdfPath,
    this.hdrPath,
    this.hdrPayload,
    this.faceSize,
    this.irradianceSize,
    this.specularSamples,
    this.diffuseSamples,
  }) : assert(hdrPath == null || hdrPayload == null, 'Provide either hdrPath or hdrPayload, not both.');

  final int environmentId;
  final String? environmentPath;
  final String? irradiancePath;
  final String? prefilteredPath;
  final String? brdfPath;
  final String? hdrPath;
  final NeuropbrTexturePayload? hdrPayload;
  final int? faceSize;
  final int? irradianceSize;
  final int? specularSamples;
  final int? diffuseSamples;

  Map<String, dynamic> toMap() => {
        'environmentId': environmentId,
        if (environmentPath != null) 'environment': environmentPath,
        if (irradiancePath != null) 'irradiance': irradiancePath,
        if (prefilteredPath != null) 'prefiltered': prefilteredPath,
        if (brdfPath != null) 'brdf': brdfPath,
        if (hdrPayload != null)
          'hdr': hdrPayload!.toMap()
        else if (hdrPath != null)
          'hdr': hdrPath,
        if (faceSize != null) 'faceSize': faceSize,
        if (irradianceSize != null) 'irradianceSize': irradianceSize,
        if (specularSamples != null) 'specularSamples': specularSamples,
        if (diffuseSamples != null) 'diffuseSamples': diffuseSamples,
      };
}
