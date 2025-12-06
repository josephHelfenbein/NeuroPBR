import Foundation
import CoreML
import UIKit
import Flutter
import UniformTypeIdentifiers
import CoreImage
import ImageIO

@available(iOS 17.0, *)
class PBRModelHandler {
    
    // Cached model instance (loading is expensive)
    // We use a static instance to ensure the model is loaded only once across the app lifecycle
    // This prevents ANE context thrashing and memory leaks from repeated loads
    private static var cachedModel: pbr_model?
    
    // Model input size (512 for memory efficiency on iPhone, upscaled to 2048 on output)
    private let modelSize = 512
    
    // CIContext for efficient image writing (thread-safe, reusable)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    
    private func getModel() throws -> pbr_model {
        if let model = PBRModelHandler.cachedModel {
            return model
        }
        
        let config = MLModelConfiguration()
        // Use all compute units - the 9 GPU preprocessing ops are negligible
        // and unified memory on iPhone means GPU vs CPU doesn't matter for memory
        config.computeUnits = .all
        
        // Allow low precision accumulation for better performance
        config.allowLowPrecisionAccumulationOnGPU = true
        
        let model = try pbr_model(configuration: config)
        PBRModelHandler.cachedModel = model
        return model
    }
    // --- MAIN METHOD: Generate PBR Textures ---
    func generatePBR(call: FlutterMethodCall, result: @escaping FlutterResult) {
        // 1. Extract arguments immediately on the main thread
        // This allows us to NOT capture 'call' in the closure, so we can release input data progressively
        guard let args = call.arguments as? [String: Any],
              let outputDir = args["outputDir"] as? String,
              let v1Data = args["view1"] as? FlutterStandardTypedData,
              let v2Data = args["view2"] as? FlutterStandardTypedData,
              let v3Data = args["view3"] as? FlutterStandardTypedData else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing outputDir or image data", details: nil))
            return
        }
        
        // Create a mutable container for inputs to allow releasing them one by one
        // We use a class to ensure reference semantics in the closure
        class InputContainer {
            var v1: FlutterStandardTypedData?
            var v2: FlutterStandardTypedData?
            var v3: FlutterStandardTypedData?
            init(v1: FlutterStandardTypedData, v2: FlutterStandardTypedData, v3: FlutterStandardTypedData) {
                self.v1 = v1
                self.v2 = v2
                self.v3 = v3
            }
        }
        let inputs = InputContainer(v1: v1Data, v2: v2Data, v3: v3Data)
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            // Hint to release any cached memory before heavy operation
            #if DEBUG
            print("[PBRModelHandler] Starting inference, triggering memory cleanup")
            #endif
            
            autoreleasepool {
                do {
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
                    
                    // Process view1
                    autoreleasepool {
                        if let data = inputs.v1?.data {
                            buffer1 = self.createBuffer(from: data)
                        }
                        inputs.v1 = nil // Release input data immediately
                    }
                    
                    // Process view2
                    autoreleasepool {
                        if let data = inputs.v2?.data {
                            buffer2 = self.createBuffer(from: data)
                        }
                        inputs.v2 = nil // Release input data immediately
                    }
                    
                    // Process view3
                    autoreleasepool {
                        if let data = inputs.v3?.data {
                            buffer3 = self.createBuffer(from: data)
                        }
                        inputs.v3 = nil // Release input data immediately
                    }
                    
                    guard let b1 = buffer1, let b2 = buffer2, let b3 = buffer3 else {
                        DispatchQueue.main.async {
                            result(FlutterError(code: "IMAGE_PROC_ERROR", message: "Failed to process input images", details: nil))
                        }
                        return
                    }

                    // 4. Run Inference
                    #if DEBUG
                    let start = CFAbsoluteTimeGetCurrent()
                    print("[PBRModelHandler] Starting prediction...")
                    #endif
                    
                    let prediction = try model.prediction(view1: b1, view2: b2, view3: b3)
                    
                    #if DEBUG
                    let duration = CFAbsoluteTimeGetCurrent() - start
                    print("[PBRModelHandler] Prediction took \(String(format: "%.2f", duration))s")
                    #endif
                    
                    // IMMEDIATELY release input buffers after inference
                    buffer1 = nil
                    buffer2 = nil
                    buffer3 = nil

                    // 5. Write outputs DIRECTLY to disk, one at a time
                    // This avoids holding all 4 outputs in memory simultaneously
                    
                    let outputPaths: [String: String] = [
                        "albedo": "\(outputDir)/albedo.png",
                        "normal": "\(outputDir)/normal.png",
                        "roughness": "\(outputDir)/roughness.png",
                        "metallic": "\(outputDir)/metallic.png"
                    ]
                    
                    // Process each output sequentially to minimize memory
                    #if DEBUG
                    let writeStart = CFAbsoluteTimeGetCurrent()
                    #endif
                    
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
                    
                    #if DEBUG
                    let writeDuration = CFAbsoluteTimeGetCurrent() - writeStart
                    print("[PBRModelHandler] Writing/Upscaling took \(String(format: "%.2f", writeDuration))s")
                    #endif

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

    // --- HELPER: Write CVPixelBuffer directly to file as PNG ---
    private func writeBufferToFile(_ buffer: CVPixelBuffer, path: String) {
        autoreleasepool {
            // Use CIContext to write directly from CVPixelBuffer to PNG
            // Model outputs at 1024x1024 (SR head does internal upscaling from 512)
            var ciImage = CIImage(cvPixelBuffer: buffer)
            
            // Upscale from 1024 to 2048 using Lanczos for final output
            if CVPixelBufferGetWidth(buffer) < 2048 {
                let scale = 2048.0 / CGFloat(CVPixelBufferGetWidth(buffer))
                
                // Use Lanczos resampling for high-quality upscaling
                let filter = CIFilter(name: "CILanczosScaleTransform")!
                filter.setValue(ciImage, forKey: kCIInputImageKey)
                filter.setValue(scale, forKey: kCIInputScaleKey)
                filter.setValue(1.0, forKey: "inputAspectRatio")
                
                if let output = filter.outputImage {
                    ciImage = output
                }
            }
            
            let url = URL(fileURLWithPath: path)
            
            do {
                // Use the shared CIContext to write PNG directly to disk (lossless, preferred for textures)
                try ciContext.writePNGRepresentation(
                    of: ciImage,
                    to: url,
                    format: .RGBA8,
                    colorSpace: ciImage.colorSpace ?? CGColorSpaceCreateDeviceRGB()
                )
            } catch {
                print("[PBRModelHandler] Error writing file: \(error)")
            }
        }
    }
    
    func clearCache() {
        PBRModelHandler.cachedModel = nil
    }
}
