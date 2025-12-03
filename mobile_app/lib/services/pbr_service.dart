import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';

class PBRService {
  static const MethodChannel _platform =
  MethodChannel('com.NeuroPBR/pbr_generator');

  /// Takes 3 image files, runs the CoreML model, and returns the PBR maps.
  Future<Map<String, Uint8List>> generatePBRMaps({
    required File view1,
    required File view2,
    required File view3,
  }) async {
    try {
      // We need to send raw data (Uint8List) to iOS
      final Uint8List bytes1 = await view1.readAsBytes();
      final Uint8List bytes2 = await view2.readAsBytes();
      final Uint8List bytes3 = await view3.readAsBytes();

      // "generatePBR" must match the guard check in Swift: call.method == "generatePBR"
      final Map<Object?, Object?> result = await _platform.invokeMethod(
          'generatePBR',
          {
            "view1": bytes1,
            "view2": bytes2,
            "view3": bytes3,
          }
      );

      // iOS sends back a Map<String, Data>. In Dart, Data becomes Uint8List.
      // We cast the generic Map to strict types for safety.
      final Map<String, Uint8List> typedResult = {};

      result.forEach((key, value) {
        if (key is String && value is Uint8List) {
          typedResult[key] = value;
        }
      });

      return typedResult;

    } on PlatformException catch (e) {
      print("Failed to generate PBR maps: '${e.message}'.");
      throw Exception("AI Generation Failed: ${e.message}");
    }
  }
}