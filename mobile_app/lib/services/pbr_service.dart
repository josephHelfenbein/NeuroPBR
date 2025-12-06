import 'dart:typed_data';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

class PBRService {
  static const MethodChannel _platform =
  MethodChannel('com.NeuroPBR/pbr_generator');
  
  // Target size for model input (512 for memory efficiency on iPhone)
  static const int _modelInputSize = 512;

  /// Resize image to target size for model input
  /// Uses high-quality Lanczos interpolation
  Future<Uint8List> _resizeImageToModelSize(File imageFile) async {
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
      width: _modelInputSize,
      height: _modelInputSize,
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
      // Read and resize input images to model size (512x512)
      // This reduces memory usage significantly for iPhone
      final Uint8List bytes1 = await _resizeImageToModelSize(view1);
      final Uint8List bytes2 = await _resizeImageToModelSize(view2);
      final Uint8List bytes3 = await _resizeImageToModelSize(view3);

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