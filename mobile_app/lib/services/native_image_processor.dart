import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';

/// Native image processor using Swift/CoreGraphics
/// Much faster than Dart's image package
class NativeImageProcessor {
  static const MethodChannel _channel = MethodChannel('com.NeuroPBR/image_processor');

  /// Process a single image: center-crop to square and resize
  /// Returns processed JPEG bytes
  static Future<Uint8List?> processImage(
    Uint8List imageData, {
    int targetSize = 2048,
    double quality = 0.92,
  }) async {
    try {
      final result = await _channel.invokeMethod('processImage', {
        'imageData': imageData,
        'targetSize': targetSize,
        'quality': quality,
      });
      return result as Uint8List?;
    } on PlatformException catch (e) {
      print('Failed to process image: ${e.message}');
      return null;
    }
  }

  /// Process a file and save result to a new file
  /// Returns the path to the processed file
  static Future<String?> processImageFile(
    String inputPath, {
    int targetSize = 2048,
    double quality = 0.92,
  }) async {
    try {
      final file = File(inputPath);
      final bytes = await file.readAsBytes();
      
      final processedBytes = await processImage(
        bytes,
        targetSize: targetSize,
        quality: quality,
      );
      
      if (processedBytes == null) return null;
      
      // Save to temp file
      final tempDir = await Directory.systemTemp.createTemp('neuropbr_');
      final outFile = File('${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}.jpg');
      await outFile.writeAsBytes(processedBytes);
      
      return outFile.path;
    } catch (e) {
      print('Error processing image file: $e');
      return null;
    }
  }

  /// Process multiple images in batch (more efficient than one-by-one)
  static Future<List<Uint8List>> processImages(
    List<Uint8List> imagesData, {
    int targetSize = 2048,
    double quality = 0.92,
  }) async {
    try {
      final result = await _channel.invokeMethod('processImages', {
        'imagesData': imagesData,
        'targetSize': targetSize,
        'quality': quality,
      });
      
      if (result is List) {
        return result.cast<Uint8List>();
      }
      return [];
    } on PlatformException catch (e) {
      print('Failed to process images: ${e.message}');
      return [];
    }
  }
}
