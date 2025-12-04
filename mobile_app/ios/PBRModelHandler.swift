import Foundation
import CoreML
import UIKit
import Flutter
import UniformTypeIdentifiers

@available(iOS 16.0, *)
class PBRModelHandler {
    
    // Cached model instance (loading is expensive)
    private var cachedModel: pbr_model?
    
    // Model input size
    private let modelSize = 2048
    
    // JPEG quality for outputs
    private let jpegQuality: CGFloat = 0.92
    
    private func getModel() throws -> pbr_model {
        if let model = cachedModel {
            return model
        }
        
        let config = MLModelConfiguration()
        // Use all compute units - the 9 GPU preprocessing ops are negligible
        // and unified memory on iPhone means GPU vs CPU doesn't matter for memory
        config.computeUnits = .all
        
        // Allow low precision accumulation for better performance
        config.allowLowPrecisionAccumulationOnGPU = true
        
        let model = try pbr_model(configuration: config)
        cachedModel = model
        return model
    }
    // --- MAIN METHOD: Generate PBR Textures ---
    func generatePBR(call: FlutterMethodCall, result: @escaping FlutterResult) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            // Hint to release any cached memory before heavy operation
            #if DEBUG
            print("[PBRModelHandler] Starting inference, triggering memory cleanup")
            #endif
            
            autoreleasepool {
                do {
                    // 1. Parse arguments and immediately extract data
                    guard let args = call.arguments as? [String: Any],
                          let outputDir = args["outputDir"] as? String else {
                        DispatchQueue.main.async {
                            result(FlutterError(code: "INVALID_ARGS", message: "Missing outputDir", details: nil))
                        }
                        return
                    }
                    
                    // Extract image data - we'll release each after creating its buffer
                    guard let v1TypedData = args["view1"] as? FlutterStandardTypedData,
                          let v2TypedData = args["view2"] as? FlutterStandardTypedData,
                          let v3TypedData = args["view3"] as? FlutterStandardTypedData else {
                        DispatchQueue.main.async {
                            result(FlutterError(code: "INVALID_ARGS", message: "Missing image data", details: nil))
                        }
                        return
                    }
                    
                    // Create output directory if needed
                    let fileManager = FileManager.default
                    if !fileManager.fileExists(atPath: outputDir) {
                        try fileManager.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
                    }

                    // 2. Load model FIRST (before creating buffers)
                    // This ensures model memory is allocated before we add input buffers
                    let model = try self.getModel()

                    // 3. Process input images SEQUENTIALLY to reduce peak memory
                    // Create buffer, then immediately release source Data
                    var buffer1: CVPixelBuffer?
                    var buffer2: CVPixelBuffer?
                    var buffer3: CVPixelBuffer?
                    
                    // Process view1 - extract data and release typed data reference
                    autoreleasepool {
                        let data = v1TypedData.data
                        buffer1 = self.createBuffer(from: data)
                    }
                    
                    // Process view2
                    autoreleasepool {
                        let data = v2TypedData.data
                        buffer2 = self.createBuffer(from: data)
                    }
                    
                    // Process view3
                    autoreleasepool {
                        let data = v3TypedData.data
                        buffer3 = self.createBuffer(from: data)
                    }
                    
                    guard let b1 = buffer1, let b2 = buffer2, let b3 = buffer3 else {
                        DispatchQueue.main.async {
                            result(FlutterError(code: "IMAGE_PROC_ERROR", message: "Failed to process input images", details: nil))
                        }
                        return
                    }

                    // 4. Run Inference
                    let prediction = try model.prediction(view1: b1, view2: b2, view3: b3)
                    
                    // IMMEDIATELY release input buffers after inference
                    buffer1 = nil
                    buffer2 = nil
                    buffer3 = nil

                    // 5. Write outputs DIRECTLY to disk, one at a time
                    // This avoids holding all 4 outputs in memory simultaneously
                    
                    let outputPaths: [String: String] = [
                        "albedo": "\(outputDir)/albedo.jpg",
                        "normal": "\(outputDir)/normal.jpg",
                        "roughness": "\(outputDir)/roughness.jpg",
                        "metallic": "\(outputDir)/metallic.jpg"
                    ]
                    
                    // Process each output sequentially to minimize memory
                    autoreleasepool {
                        self.writeBufferToFile(prediction.albedo, path: outputPaths["albedo"]!)
                    }
                    autoreleasepool {
                        self.writeBufferToFile(prediction.normal, path: outputPaths["normal"]!)
                    }
                    autoreleasepool {
                        self.writeBufferToFile(prediction.roughness, path: outputPaths["roughness"]!)
                    }
                    autoreleasepool {
                        self.writeBufferToFile(prediction.metallic, path: outputPaths["metallic"]!)
                    }

                    // 6. Return success with paths (no image data in memory!)
                    DispatchQueue.main.async {
                        result(outputPaths)
                    }

                } catch {
                    DispatchQueue.main.async {
                        result(FlutterError(code: "INFERENCE_ERROR", message: error.localizedDescription, details: nil))
                    }
                }
            }
        }
    }

    // --- HELPER: Data -> CVPixelBuffer ---
    private func createBuffer(from imageData: Data) -> CVPixelBuffer? {
        autoreleasepool {
            guard let image = UIImage(data: imageData),
                  let cgImage = image.cgImage else { return nil }
            
            // Create pixel buffer with Metal compatibility for efficient Neural Engine transfer
            // Using IOSurface allows zero-copy sharing between CPU, GPU, and Neural Engine
            let options: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
                kCVPixelBufferMetalCompatibilityKey: true  // Enable Metal compatibility
            ]
            
            var pxBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault,
                modelSize, modelSize,
                kCVPixelFormatType_32BGRA,
                options as CFDictionary,
                &pxBuffer
            )
            
            guard status == kCVReturnSuccess, let buffer = pxBuffer else { return nil }

            CVPixelBufferLockBaseAddress(buffer, [])
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            
            guard let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: modelSize,
                height: modelSize,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            ) else { return nil }
            
            // Use default interpolation (faster than high quality, still good enough)
            context.interpolationQuality = .default
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: modelSize, height: modelSize))

            return buffer
        }
    }

    // --- HELPER: Write CVPixelBuffer directly to file as JPEG ---
    private func writeBufferToFile(_ buffer: CVPixelBuffer, path: String) {
        autoreleasepool {
            // Lock buffer for reading
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            
            let width = CVPixelBufferGetWidth(buffer)
            let height = CVPixelBufferGetHeight(buffer)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            
            guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return }
            
            // Create CGImage directly from buffer data
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            ) else { return }
            
            guard let cgImage = context.makeImage() else { return }
            
            // Write JPEG directly to file using ImageIO (memory efficient)
            let url = URL(fileURLWithPath: path)
            guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else { return }
            
            let options: [CFString: Any] = [
                kCGImageDestinationLossyCompressionQuality: jpegQuality
            ]
            
            CGImageDestinationAddImage(destination, cgImage, options as CFDictionary)
            CGImageDestinationFinalize(destination)
        }
    }
    
    func clearCache() {
        cachedModel = nil
    }
}
