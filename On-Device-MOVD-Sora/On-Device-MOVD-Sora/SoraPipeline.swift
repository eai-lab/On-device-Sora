import Foundation
import CoreML
import Accelerate
import Tokenizers
import Hub

struct PackageURLs {
    public let stditURL: URL
    public let decoderURL: URL
    public let configT5URL: URL
    public let dataT5URL: URL
    public let embedURL: URL
    public let finalNormURL: URL
  
    public init(resourcesAt baseURL: URL) {
        stditURL = baseURL.appending(path: "stdit3_part1.mlmodelc")
        decoderURL = baseURL.appending(path: "vae_spatial_part1.mlmodelc")
        configT5URL = baseURL.appending(path: "tokenizer_config.json")
        dataT5URL = baseURL.appending(path: "tokenizer.json")
        embedURL = baseURL.appending(path: "t5embed-tokens.mlmodelc")
        finalNormURL = baseURL.appending(path: "t5final-layer-norm.mlmodelc")
    }
}




public struct SoraPipeline {
  // need to initialize the required models. ex) stdit, vae and so on.
  let TextEncodingT5: TextEncoding?
  let STDiT: STDiT?
  let VAE: VAEDecoder?
  let Converter: Tensor2Vid?

  init(resourcesAt baseURL: URL, videoConverter converter: Tensor2Vid ) throws {
    let urls = PackageURLs(resourcesAt: baseURL)
    Converter = converter
    // initialize Models for Text Encoding
    if FileManager.default.fileExists(atPath: urls.configT5URL.path),
    FileManager.default.fileExists(atPath: urls.dataT5URL.path),
    FileManager.default.fileExists(atPath: urls.embedURL.path),
    FileManager.default.fileExists(atPath: urls.finalNormURL.path)
    {
      let config = MLModelConfiguration()
      config.computeUnits = .cpuAndGPU
      let tokenizerT5 = try PreTrainedTokenizer(tokenizerConfig: Config(fileURL: urls.configT5URL), tokenizerData: Config(fileURL: urls.dataT5URL))
      let embedLayer = ManagedMLModel(modelURL: urls.embedURL, config: config)
      let finalNormLayer = ManagedMLModel(modelURL: urls.finalNormURL, config: config)
      var DivT5s: [ManagedMLModel] = []
      for i in 0...23 {
          let T5BlockURL = baseURL.appending(path: "t5block-layer\(i).mlmodelc")
          DivT5s.append(ManagedMLModel(modelURL: T5BlockURL, config: config))
      }
      TextEncodingT5 = TextEncoding(tokenizer: tokenizerT5, DivT5s: DivT5s, embed: embedLayer, finalNorm: finalNormLayer)
      } else {
      TextEncodingT5 = nil
    }
    
    // initialize Models for STDit
    if FileManager.default.fileExists(atPath: urls.stditURL.path) {
      // To do: STDit Model
      let config_stdit = MLModelConfiguration()
      config_stdit.computeUnits = .cpuAndGPU
      let part1 = ManagedMLModel(modelURL: baseURL.appending(path: "stdit3_part1.mlmodelc"), config: config_stdit)
      var spatialAndTemporalBlocks: [ManagedMLModel] = []
      var STDiTTDTMBLocks: [ManagedMLModel] = []
      for i in 0...27 {
          let spatialsBlockURL = baseURL.appending(path: "stdit3_ST_\(i).mlmodelc")
          spatialAndTemporalBlocks.append(ManagedMLModel(modelURL: spatialsBlockURL, config: config_stdit))
          if FileManager.default.fileExists(atPath: baseURL.appending(path: "stdit3_TDTM_\(i).mlmodelc").path()) {
            let TDTMBlock = baseURL.appending(path: "stdit3_TDTM_\(i).mlmodelc")
            STDiTTDTMBLocks.append(ManagedMLModel(modelURL: TDTMBlock, config: config_stdit))
          }
      }
      let part2 = ManagedMLModel(modelURL: baseURL.appending(path: "stdit3_part2.mlmodelc"), config: config_stdit)
        STDiT = On_Device_MOVD_Sora.STDiT(part1: part1, spatialAndTemporals: spatialAndTemporalBlocks, STDiTTDTMBLocks: STDiTTDTMBLocks, part2: part2)
      } else {
      STDiT = nil
    }
    
    // initialize Models for VAE
    if FileManager.default.fileExists(atPath: urls.decoderURL.path) {
    // To do: VAE for decoding video
    let config_vae = MLModelConfiguration()
    config_vae.computeUnits = .cpuOnly
    VAE = VAEDecoder(config: config_vae)
    } else {
    VAE = nil
    }
  }

  func sample(prompt: String, logdir: URL, seed: Int, step: Int, mergeStep: Int,numLpltarget: Int, isBase: Bool, isLPL: Bool, isTDTM: Bool, isCI: Bool, isDL: Bool) {
    // To do: make the sample process
    
    Task(priority: .high) {
      do {
          let filename = "sample-\(prompt).mp4"
          print(prompt)
          let startTotalTime = DispatchTime.now()
      
          // =========== T5 ===========
          guard let ids = try TextEncodingT5?.tokenize(prompt) else {
              print("Error: Can't tokenize")
              return
          }
          guard let resultEncoding = try TextEncodingT5?.encode(ids: ids) else {
              print("Error: Can't Encoding")
              return
          }
          print("Done T5 Encoding")
          // ==========================

          // =========== STDiT ===========
          let additionalArgs: [String: MLTensor] = [:]
          let vaeOutChannels = 4
          let latentsize = (20, isBase ? 32 : 24, isBase ? 32 : 24)
          let width = isBase ? 256.0 : 192.0
          let height = isBase ? 256.0 : 192.0
          let fps = 24.0
          let resolution = width * height
          let z = await MLTensor(randomNormal: [1, vaeOutChannels, latentsize.0, latentsize.1, latentsize.2],seed: UInt64(seed), scalarType: Float32.self).shapedArray(of: Float32.self)
          let mask = await MLTensor(ones: [latentsize.0], scalarType: Float32.self).shapedArray(of: Float32.self)
          let dynamicSize = getDynamicSize(latentSize: latentsize)
          let modelArgs = ["y": resultEncoding.encoderHiddenStates, "mask": resultEncoding.masks, "fps" : MLShapedArray<Float32>(arrayLiteral: Float32(fps)) , "width": MLShapedArray<Float32>(arrayLiteral: Float32(width)), "height":MLShapedArray<Float32>(arrayLiteral: Float32(height)), "padH": MLShapedArray<Float32>(arrayLiteral: Float32(dynamicSize.0)), "padW": MLShapedArray<Float32>(arrayLiteral: Float32(dynamicSize.1))]
          let rflowInput = RFLOWInput(model: STDiT!, modelArgs: modelArgs, z: z, mask: mask, additionalArgs: additionalArgs, resolution: resolution)
          
          let rflow = RFLOW(numSamplingsteps: step, cfgScale: 7.0, seed: UInt64(seed), idsCount: ids.count+1, attnSize: dynamicSize.2, mergeStep: mergeStep, numLpltarget: numLpltarget, isLPL: isLPL, isTDTM: isTDTM, isCI: isCI, isDL: isDL)
          let resultSTDit = await rflow.sample(rflowInput: rflowInput, yNull: resultEncoding.yNull).shapedArray(of: Float32.self)
          print(resultSTDit.shape)

          // ==========================

          // =========== VAE ===========
          guard let resultDecoding = try await VAE?.decode(latentVars: resultSTDit) else {
              print("Error: Can't Decode")
              return
          }
          print("Decoding-shape:")
          print(resultDecoding.shape)
          // ==========================

          let _ = await Converter!.convertToVideo(multiArray: resultDecoding, logdir: logdir, filename: filename)
          let endTotalTime = DispatchTime.now()
          let elapsedTotalTime = endTotalTime.uptimeNanoseconds - startTotalTime.uptimeNanoseconds
          print("Total Running Time: \(Double(elapsedTotalTime) / 1000000000) seconds")
      } catch {
          print("Error: Can't make sample.")
          print(error)
        }
      }
    }
}

extension SoraPipeline {
    func getDynamicSize(latentSize: (Int, Int, Int)) -> (Int, Int, Int) {
        var H = latentSize.1
        var W = latentSize.2
        
        if H % 2 != 0 {
            H = 2 - H % 2
        } else {
            H = 0
        }
        
        if W % 2 != 0 {
            W = 2 - W % 2
        } else {
            W = 0
        }
        
        let T = latentSize.0 * (latentSize.1 + H) * (latentSize.2 + W) / 4
        return (H, W, T)
    }
}


//extension SoraPipeline {
//  private func prepareMultiResolutionInfo(imageSize: [Int], numFrames: Int, fps: Int) -> [String : MLTensor] {
//    let fps = MLTensor([Float32(numFrames > 1 ? fps : 120)])
//    let height = MLTensor([Float32(imageSize[0])])
//    let width = MLTensor([Float32(imageSize[1])])
//    let numFrames = MLTensor([Float32(numFrames)])
//    return ["fps": fps, "height": height,"width": width, "numFrames": numFrames]
//  }
//}
