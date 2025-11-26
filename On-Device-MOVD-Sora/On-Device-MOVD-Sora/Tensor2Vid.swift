import Foundation
import CoreML
import AVFoundation
import VideoToolbox
import SwiftUI

@MainActor
final class Tensor2Vid: ObservableObject {
    @Published var videoURL: URL?

    func convertToVideo(multiArray: MLMultiArray, logdir: URL, filename: String) async -> URL? {
        let frameCount = multiArray.shape[2].intValue
        let channels = multiArray.shape[1].intValue
        let height = multiArray.shape[3].intValue
        let width = multiArray.shape[4].intValue
        
        guard channels == 3 else {
        print("Invalid number of channels. Expected 3, got \(channels)")
            return nil
        }
      
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("sample.mp4")
        let videoWriter: AVAssetWriter
        do {
        videoWriter = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        } catch {
        print("Failed to create AVAssetWriter: \(error)")
            return nil
        }
        
        let videoSettings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height
        ]
        
        let videoWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoWriterInput, sourcePixelBufferAttributes: nil)
        
        videoWriter.add(videoWriterInput)
        videoWriter.startWriting()
        videoWriter.startSession(atSourceTime: .zero)
        
        let frameDuration = CMTimeMake(value: 1, timescale: 30)
        var frameTime = CMTime.zero
        
        for frameIndex in 0..<frameCount {
                
          let pixelBuffer = createPixelBuffer(from: multiArray, frameIndex: frameIndex, width: width, height: height)
          
          while !videoWriterInput.isReadyForMoreMediaData {
          Thread.sleep(forTimeInterval: 0.1)
          }
                
          if adaptor.append(pixelBuffer!, withPresentationTime: frameTime) {
            frameTime = CMTimeAdd(frameTime, frameDuration)
            } else {
            print("Failed to append pixel buffer for frame \(frameIndex)")
            }
            
        }
        
        videoWriterInput.markAsFinished()
        await videoWriter.finishWriting()
        
        await MainActor.run {
            self.videoURL = outputURL
        }
        print("Video saved to: \(outputURL.path)")
        
        return outputURL
    }

    private func createPixelBuffer(from multiArray: MLMultiArray, frameIndex: Int, width: Int, height: Int) -> CVPixelBuffer? {
      
        var pixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width,
                                         height,
                                         kCVPixelFormatType_32BGRA,
                                         attrs,
                                         &pixelBuffer)

        guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * 4
                guard let pixelOffsetPointer = pixelData?.advanced(by: pixelOffset) else { continue }
                
                let pixel = pixelOffsetPointer.bindMemory(to: UInt8.self, capacity: 4)
                let r = normalizeTensorValue(multiArray[[0, 0, frameIndex, y, x] as [NSNumber]].floatValue, minValue: -1.0, maxValue: 1.0)
                let g = normalizeTensorValue(multiArray[[0, 1, frameIndex, y, x] as [NSNumber]].floatValue, minValue: -1.0, maxValue: 1.0)
                let b = normalizeTensorValue(multiArray[[0, 2, frameIndex, y, x] as [NSNumber]].floatValue, minValue: -1.0, maxValue: 1.0)
                
                pixel[0] = b
                pixel[1] = g
                pixel[2] = r
                pixel[3] = 255  // Alpha channel
            }
        }

        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
    }
    
    private func normalizeTensorValue(_ value: Float, minValue: Float, maxValue: Float) -> UInt8 {
        
        let normalized = (value - minValue) / (maxValue - minValue)
        return UInt8(max(0, min(255, normalized * 255)))
    }
}
