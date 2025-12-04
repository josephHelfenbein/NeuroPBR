import Foundation
import UIKit
import Flutter

/// High-performance native image processor for NeuroPBR
/// Crops to square and resizes to target size using GPU-accelerated CoreGraphics
class ImageProcessor {
    
    static let shared = ImageProcessor()
    
    private init() {}
    
    /// Process image: center-crop to square and resize to targetSize
    /// Returns JPEG data at specified quality
    func processImage(data: Data, targetSize: Int = 2048, jpegQuality: CGFloat = 0.92) -> Data? {
        autoreleasepool {
            guard let image = UIImage(data: data),
                  let cgImage = image.cgImage else {
                return nil
            }
            
            let width = cgImage.width
            let height = cgImage.height
            
            // Calculate center crop to square
            let minDim = min(width, height)
            let cropX = (width - minDim) / 2
            let cropY = (height - minDim) / 2
            let cropRect = CGRect(x: cropX, y: cropY, width: minDim, height: minDim)
            
            // Crop
            guard let croppedCGImage = cgImage.cropping(to: cropRect) else {
                return nil
            }
            
            // Resize using CoreGraphics (GPU accelerated)
            let targetCGSize = CGSize(width: targetSize, height: targetSize)
            
            UIGraphicsBeginImageContextWithOptions(targetCGSize, true, 1.0)
            defer { UIGraphicsEndImageContext() }
            
            guard let context = UIGraphicsGetCurrentContext() else {
                return nil
            }
            
            // High quality interpolation
            context.interpolationQuality = .high
            
            // Flip coordinate system for CGImage
            context.translateBy(x: 0, y: CGFloat(targetSize))
            context.scaleBy(x: 1, y: -1)
            
            // Draw the cropped image scaled to target size
            context.draw(croppedCGImage, in: CGRect(origin: .zero, size: targetCGSize))
            
            guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
                return nil
            }
            
            // Encode as JPEG
            return resizedImage.jpegData(compressionQuality: jpegQuality)
        }
    }
}

/// Flutter Method Channel handler for image processing
class ImageProcessorPlugin {
    
    static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "com.NeuroPBR/image_processor",
            binaryMessenger: registrar.messenger()
        )
        
        channel.setMethodCallHandler { call, result in
            switch call.method {
            case "processImage":
                handleProcessImage(call: call, result: result)
            case "processImages":
                handleProcessImages(call: call, result: result)
            default:
                result(FlutterMethodNotImplemented)
            }
        }
    }
    
    /// Process a single image
    private static func handleProcessImage(call: FlutterMethodCall, result: @escaping FlutterResult) {
        DispatchQueue.global(qos: .userInitiated).async {
            guard let args = call.arguments as? [String: Any],
                  let imageData = (args["imageData"] as? FlutterStandardTypedData)?.data else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "INVALID_ARGS", message: "Missing imageData", details: nil))
                }
                return
            }
            
            let targetSize = args["targetSize"] as? Int ?? 2048
            let quality = args["quality"] as? Double ?? 0.92
            
            if let processedData = ImageProcessor.shared.processImage(
                data: imageData,
                targetSize: targetSize,
                jpegQuality: CGFloat(quality)
            ) {
                DispatchQueue.main.async {
                    result(FlutterStandardTypedData(bytes: processedData))
                }
            } else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "PROCESS_ERROR", message: "Failed to process image", details: nil))
                }
            }
        }
    }
    
    /// Process multiple images in batch
    private static func handleProcessImages(call: FlutterMethodCall, result: @escaping FlutterResult) {
        DispatchQueue.global(qos: .userInitiated).async {
            guard let args = call.arguments as? [String: Any],
                  let imagesData = args["imagesData"] as? [FlutterStandardTypedData] else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "INVALID_ARGS", message: "Missing imagesData", details: nil))
                }
                return
            }
            
            let targetSize = args["targetSize"] as? Int ?? 2048
            let quality = args["quality"] as? Double ?? 0.92
            
            var processedImages: [FlutterStandardTypedData] = []
            
            for imageTypedData in imagesData {
                autoreleasepool {
                    if let processedData = ImageProcessor.shared.processImage(
                        data: imageTypedData.data,
                        targetSize: targetSize,
                        jpegQuality: CGFloat(quality)
                    ) {
                        processedImages.append(FlutterStandardTypedData(bytes: processedData))
                    }
                }
            }
            
            DispatchQueue.main.async {
                result(processedImages)
            }
        }
    }
}
