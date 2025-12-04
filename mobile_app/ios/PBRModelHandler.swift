import Foundation
import CoreML
import UIKit
import Flutter

@available(iOS 16.0, *)
class PBRModelHandler {
    
    // Shared CIContext to avoid repeated allocations
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    
    // Cached model instance (loading is expensive)
    private var cachedModel: pbr_model?
    
    private func getModel() throws -> pbr_model {
        if let model = cachedModel {
            return model
        }
        
        let config = MLModelConfiguration()
        // Use CPU + Neural Engine, avoid GPU to reduce memory pressure
        config.computeUnits = .cpuAndNeuralEngine
        
        let model = try pbr_model(configuration: config)
        cachedModel = model
        return model
    }
    
    // MAIN FUNCTION CALLED BY FLUTTER
    func generatePBR(call: FlutterMethodCall, result: @escaping FlutterResult) {
        // Use autoreleasepool to ensure memory is freed promptly
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            autoreleasepool {
                do {
                    // 1. Parse arguments (raw bytes from Flutter)
                    guard let args = call.arguments as? [String: Any],
                          let v1Data = (args["view1"] as? FlutterStandardTypedData)?.data,
                          let v2Data = (args["view2"] as? FlutterStandardTypedData)?.data,
                          let v3Data = (args["view3"] as? FlutterStandardTypedData)?.data else {
                        result(FlutterError(code: "INVALID_ARGS", message: "Missing image data", details: nil))
                        return
                    }

                    // 2. Load Core ML model (cached)
                    let model = try self.getModel()

                    // 3. Preprocess images one at a time to reduce peak memory
                    // Process and release each buffer sequentially
                    var buffer1: CVPixelBuffer?
                    var buffer2: CVPixelBuffer?
                    var buffer3: CVPixelBuffer?
                    
                    autoreleasepool {
                        buffer1 = self.buffer(from: v1Data)
                    }
                    autoreleasepool {
                        buffer2 = self.buffer(from: v2Data)
                    }
                    autoreleasepool {
                        buffer3 = self.buffer(from: v3Data)
                    }
                    
                    guard let b1 = buffer1, let b2 = buffer2, let b3 = buffer3 else {
                        result(FlutterError(code: "IMAGE_PROC_ERROR", message: "Failed to process input images", details: nil))
                        return
                    }

                    // 4. Run Inference
                    let prediction = try model.prediction(view1: b1, view2: b2, view3: b3)
                    
                    // Clear input buffers immediately after inference
                    buffer1 = nil
                    buffer2 = nil
                    buffer3 = nil

                    // 5. Extract outputs one at a time to reduce peak memory
                    var albedoData: Data?
                    var normalData: Data?
                    var roughnessData: Data?
                    var metallicData: Data?
                    
                    autoreleasepool {
                        albedoData = self.data(from: prediction.albedo)
                    }
                    autoreleasepool {
                        normalData = self.data(from: prediction.normal)
                    }
                    autoreleasepool {
                        roughnessData = self.data(from: prediction.roughness)
                    }
                    autoreleasepool {
                        metallicData = self.data(from: prediction.metallic)
                    }

                    // 6. Return to Flutter
                    let response: [String: Any] = [
                        "albedo": albedoData ?? Data(),
                        "normal": normalData ?? Data(),
                        "roughness": roughnessData ?? Data(),
                        "metallic": metallicData ?? Data()
                    ]
                    
                    DispatchQueue.main.async {
                        result(response)
                    }

                } catch {
                    DispatchQueue.main.async {
                        result(FlutterError(code: "INFERENCE_ERROR", message: error.localizedDescription, details: nil))
                    }
                }
            }
        }
    }

    // --- HELPER: Resizes and converts Data -> CVPixelBuffer ---
    private func buffer(from imageData: Data) -> CVPixelBuffer? {
        autoreleasepool {
            guard let image = UIImage(data: imageData),
                  let cgImage = image.cgImage else { return nil }

            // Use IOSurface-backed buffer for better memory management
            let options: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
            ]

            var pxBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(kCFAllocatorDefault,
                2048, 2048,
                kCVPixelFormatType_32ARGB,
                options as CFDictionary,
                &pxBuffer)

            guard status == kCVReturnSuccess, let buffer = pxBuffer else { return nil }

            CVPixelBufferLockBaseAddress(buffer, [])
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            
            let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                width: 2048,
                height: 2048,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

            context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: 2048, height: 2048))

            return buffer
        }
    }

    // --- HELPER: CVPixelBuffer -> Data (JPEG for smaller memory footprint) ---
    private func data(from buffer: CVPixelBuffer) -> Data? {
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: buffer)
            if let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) {
                // Use PNG for quality (lossless output maps)
                return UIImage(cgImage: cgImage).pngData()
            }
            return nil
        }
    }
    
    // Call this to free memory if needed
    func clearCache() {
        cachedModel = nil
    }
}
