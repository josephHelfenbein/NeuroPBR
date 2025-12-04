import Foundation
import CoreML
import UIKit
import Flutter
import Accelerate

@available(iOS 16.0, *)
class PBRModelHandler {
    
    // Shared CIContext with Metal for GPU-accelerated rendering
    private let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() {
            return CIContext(mtlDevice: device, options: [.cacheIntermediates: false])
        }
        return CIContext(options: [.useSoftwareRenderer: false, .cacheIntermediates: false])
    }()
    
    // Cached model instance (loading is expensive)
    private var cachedModel: pbr_model?
    
    // Pre-allocated pixel buffer pool for reuse
    private var bufferPool: CVPixelBufferPool?
    
    // Model input size
    private let modelSize = 2048
    
    // JPEG quality for outputs (0.85 is good balance of quality/speed)
    private let jpegQuality: CGFloat = 0.90
    
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
    
    // Get or create a reusable pixel buffer pool
    private func getBufferPool() -> CVPixelBufferPool? {
        if let pool = bufferPool {
            return pool
        }
        
        let poolAttributes: [CFString: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey: 3
        ]
        
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferWidthKey: modelSize,
            kCVPixelBufferHeightKey: modelSize,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                poolAttributes as CFDictionary,
                                bufferAttributes as CFDictionary,
                                &pool)
        bufferPool = pool
        return pool
    }
    
    // MAIN FUNCTION CALLED BY FLUTTER
    func generatePBR(call: FlutterMethodCall, result: @escaping FlutterResult) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            autoreleasepool {
                do {
                    // 1. Parse arguments (raw bytes from Flutter)
                    guard let args = call.arguments as? [String: Any],
                          let v1Data = (args["view1"] as? FlutterStandardTypedData)?.data,
                          let v2Data = (args["view2"] as? FlutterStandardTypedData)?.data,
                          let v3Data = (args["view3"] as? FlutterStandardTypedData)?.data else {
                        DispatchQueue.main.async {
                            result(FlutterError(code: "INVALID_ARGS", message: "Missing image data", details: nil))
                        }
                        return
                    }

                    // 2. Load Core ML model (cached)
                    let model = try self.getModel()

                    // 3. Process all three images in parallel for speed
                    var buffer1: CVPixelBuffer?
                    var buffer2: CVPixelBuffer?
                    var buffer3: CVPixelBuffer?
                    
                    let bufferGroup = DispatchGroup()
                    let bufferQueue = DispatchQueue(label: "buffer.processing", attributes: .concurrent)
                    
                    bufferGroup.enter()
                    bufferQueue.async {
                        autoreleasepool {
                            buffer1 = self.buffer(from: v1Data)
                        }
                        bufferGroup.leave()
                    }
                    
                    bufferGroup.enter()
                    bufferQueue.async {
                        autoreleasepool {
                            buffer2 = self.buffer(from: v2Data)
                        }
                        bufferGroup.leave()
                    }
                    
                    bufferGroup.enter()
                    bufferQueue.async {
                        autoreleasepool {
                            buffer3 = self.buffer(from: v3Data)
                        }
                        bufferGroup.leave()
                    }
                    
                    bufferGroup.wait()
                    
                    guard let b1 = buffer1, let b2 = buffer2, let b3 = buffer3 else {
                        DispatchQueue.main.async {
                            result(FlutterError(code: "IMAGE_PROC_ERROR", message: "Failed to process input images", details: nil))
                        }
                        return
                    }

                    // 4. Run Inference
                    let prediction = try model.prediction(view1: b1, view2: b2, view3: b3)
                    
                    // Clear input buffers immediately after inference
                    buffer1 = nil
                    buffer2 = nil
                    buffer3 = nil

                    // 5. Extract all outputs in parallel for speed
                    var albedoData: Data?
                    var normalData: Data?
                    var roughnessData: Data?
                    var metallicData: Data?
                    
                    let outputGroup = DispatchGroup()
                    let outputQueue = DispatchQueue(label: "output.processing", attributes: .concurrent)
                    
                    outputGroup.enter()
                    outputQueue.async {
                        autoreleasepool {
                            albedoData = self.data(from: prediction.albedo)
                        }
                        outputGroup.leave()
                    }
                    
                    outputGroup.enter()
                    outputQueue.async {
                        autoreleasepool {
                            normalData = self.data(from: prediction.normal)
                        }
                        outputGroup.leave()
                    }
                    
                    outputGroup.enter()
                    outputQueue.async {
                        autoreleasepool {
                            roughnessData = self.data(from: prediction.roughness)
                        }
                        outputGroup.leave()
                    }
                    
                    outputGroup.enter()
                    outputQueue.async {
                        autoreleasepool {
                            metallicData = self.data(from: prediction.metallic)
                        }
                        outputGroup.leave()
                    }
                    
                    outputGroup.wait()

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

    // --- HELPER: Fast Data -> CVPixelBuffer using CGImage directly ---
    private func buffer(from imageData: Data) -> CVPixelBuffer? {
        autoreleasepool {
            // Use CGImageSource for faster decoding than UIImage
            guard let imageSource = CGImageSourceCreateWithData(imageData as CFData, nil),
                  let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, [
                    kCGImageSourceShouldCache: false,
                    kCGImageSourceShouldAllowFloat: false
                  ] as CFDictionary) else {
                // Fallback to UIImage if CGImageSource fails
                guard let image = UIImage(data: imageData),
                      let cgImage = image.cgImage else { return nil }
                return createBuffer(from: cgImage)
            }
            
            return createBuffer(from: cgImage)
        }
    }
    
    private func createBuffer(from cgImage: CGImage) -> CVPixelBuffer? {
        // Try to get buffer from pool first (reuses memory)
        var pxBuffer: CVPixelBuffer?
        
        if let pool = getBufferPool() {
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pxBuffer)
        }
        
        // Fallback to direct creation if pool fails
        if pxBuffer == nil {
            let options: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
            ]
            
            CVPixelBufferCreate(kCFAllocatorDefault,
                modelSize, modelSize,
                kCVPixelFormatType_32BGRA,
                options as CFDictionary,
                &pxBuffer)
        }
        
        guard let buffer = pxBuffer else { return nil }

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
        
        // High-quality interpolation for resize
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: modelSize, height: modelSize))

        return buffer
    }

    // --- HELPER: CVPixelBuffer -> Data using fast JPEG encoding ---
    private func data(from buffer: CVPixelBuffer) -> Data? {
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: buffer)
            
            // Use JPEG for much faster encoding (5-10x faster than PNG)
            // Quality 0.90 is visually nearly identical to PNG
            if let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) {
                // CIContext.jpegRepresentation doesn't take options dict, quality is set at context level
                // Use UIImage conversion for JPEG with quality control
                if let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) {
                    return UIImage(cgImage: cgImage).jpegData(compressionQuality: jpegQuality)
                }
            }
            
            // Fallback to PNG if JPEG fails
            if let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) {
                return UIImage(cgImage: cgImage).pngData()
            }
            return nil
        }
    }
    
    // Call this to free memory if needed
    func clearCache() {
        cachedModel = nil
        bufferPool = nil
    }
}
