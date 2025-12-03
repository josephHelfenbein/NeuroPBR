import Foundation
import CoreML
import UIKit

@available(iOS 15.0, *)
class PBRModelHandler {

    // Xcode auto-generates the class 'pbr_model' from the file 'pbr_model.mlpackage'
    let model: pbr_model

    init() {
        do {
            let config = MLModelConfiguration()
            // Force use of Neural Engine (ANE) and GPU
            config.computeUnits = .all
            self.model = try pbr_model(configuration: config)
        } catch {
            fatalError("Failed to load Core ML model: \(error)")
        }
    }

    // MAIN FUNCTION CALLED BY FLUTTER
    func generateMaps(view1Data: Data, view2Data: Data, view3Data: Data) throws -> [String: Data] {

        // 1. Convert raw bytes (PNG/JPG) to CVPixelBuffers sized 2048x2048
        guard let buffer1 = buffer(from: view1Data),
        let buffer2 = buffer(from: view2Data),
        let buffer3 = buffer(from: view3Data) else {
            throw NSError(domain: "PBRModelError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to resize/convert input images"])
        }

        // 2. Run Prediction
        // The argument names 'view1', 'view2' come from the Python conversion script
        let output = try model.prediction(view1: buffer1, view2: buffer2, view3: buffer3)

        // 3. Convert Output Buffers back to PNG Data
        let albedoData = data(from: output.albedo)
        let normalData = data(from: output.normal)
        let roughnessData = data(from: output.roughness)
        let metallicData = data(from: output.metallic)

        // 4. Return Dictionary
        return [
            "albedo": albedoData ?? Data(),
            "normal": normalData ?? Data(),
            "roughness": roughnessData ?? Data(),
            "metallic": metallicData ?? Data()
        ]
    }

    // --- HELPER: Resizes and converts Data -> CVPixelBuffer ---
    private func buffer(from imageData: Data) -> CVPixelBuffer? {
        guard let image = UIImage(data: imageData),
        let cgImage = image.cgImage else { return nil }

        let options: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        var pxBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
            2048, 2048, // MODEL SIZE
            kCVPixelFormatType_32ARGB,
            options as CFDictionary,
            &pxBuffer)

        guard status == kCVReturnSuccess, let buffer = pxBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
            width: 2048,
            height: 2048,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        // Draw the image into the context (This handles resizing automatically)
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: 2048, height: 2048))
        CVPixelBufferUnlockBaseAddress(buffer, [])

        return buffer
    }

    // --- HELPER: CVPixelBuffer -> Data (PNG) ---
    private func data(from buffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            return UIImage(cgImage: cgImage).pngData()
        }
        return nil
    }
}