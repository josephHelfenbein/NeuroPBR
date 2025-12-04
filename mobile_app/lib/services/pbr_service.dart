import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';

class PBRService {
  static const MethodChannel _platform =
  MethodChannel('com.NeuroPBR/pbr_generator');

  /// Takes 3 image files and an output directory, runs the CoreML model,
  /// and writes PBR maps directly to disk. Returns the file paths.
  Future<Map<String, String>> generatePBRMaps({
    required File view1,
    required File view2,
    required File view3,
    required String outputDir,
  }) async {
    try {
      // Read input images
      final Uint8List bytes1 = await view1.readAsBytes();
      final Uint8List bytes2 = await view2.readAsBytes();
      final Uint8List bytes3 = await view3.readAsBytes();

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