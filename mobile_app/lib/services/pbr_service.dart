import 'dart:typed_data';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

class PBRService {
  static const MethodChannel _platform =
  MethodChannel('com.NeuroPBR/pbr_generator');
  
  // Cached model input size (read from model on first call)
  static int? _cachedModelInputSize;
  
  // Default fallback size if model query fails
  static const int _defaultModelInputSize = 512;

  /// Get the expected input size from the CoreML model
  /// Caches the result for subsequent calls
  Future<int> getModelInputSize() async {
    if (_cachedModelInputSize != null) {
      return _cachedModelInputSize!;
    }
    
    try {
      final int size = await _platform.invokeMethod('getModelInputSize');
      _cachedModelInputSize = size;
      return size;
    } on PlatformException catch (e) {
      print("Failed to get model input size: '${e.message}'. Using default.");
      return _defaultModelInputSize;
    }
  }

  /// Resize image to target size for model input
  /// Uses high-quality Lanczos interpolation
  Future<Uint8List> _resizeImageToModelSize(File imageFile, int targetSize) async {
    final Uint8List bytes = await imageFile.readAsBytes();
    final img.Image? image = img.decodeImage(bytes);
    if (image == null) {
      throw Exception('Failed to decode image: ${imageFile.path}');
    }
    
    // Center crop to square first
    final int minDim = image.width < image.height ? image.width : image.height;
    final int cropX = (image.width - minDim) ~/ 2;
    final int cropY = (image.height - minDim) ~/ 2;
    final img.Image cropped = img.copyCrop(image, x: cropX, y: cropY, width: minDim, height: minDim);
    
    // Resize to model size using cubic interpolation
    final img.Image resized = img.copyResize(
      cropped,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.cubic,
    );
    
    // Encode as PNG (lossless, preferred for texture workflows)
    return Uint8List.fromList(img.encodePng(resized));
  }

  /// Takes 3 image files and an output directory, runs the CoreML model,
  /// and writes PBR maps directly to disk. Returns the file paths.
  Future<Map<String, String>> generatePBRMaps({
    required File view1,
    required File view2,
    required File view3,
    required String outputDir,
  }) async {
    try {
      // Get the model's expected input size
      final int modelInputSize = await getModelInputSize();
      
      // Read and resize input images to model size
      // This reduces memory usage significantly for iPhone
      final Uint8List bytes1 = await _resizeImageToModelSize(view1, modelInputSize);
      final Uint8List bytes2 = await _resizeImageToModelSize(view2, modelInputSize);
      final Uint8List bytes3 = await _resizeImageToModelSize(view3, modelInputSize);

      // Call native - it will write directly to disk and return paths
      final Map<Object?, Object?> result = await _platform.invokeMethod(
          'generatePBR',
          {
            "view1": bytes1,
            "view2": bytes2,
            "view3": bytes3,
            "outputDir": outputDir,
          }
      );

      // Convert to typed map of paths
      final Map<String, String> paths = {};
      result.forEach((key, value) {
        if (key is String && value is String) {
          paths[key] = value;
        }
      });

      return paths;

    } on PlatformException catch (e) {
      print("Failed to generate PBR maps: '${e.message}'.");
      throw Exception("AI Generation Failed: ${e.message}");
    }
  }
}