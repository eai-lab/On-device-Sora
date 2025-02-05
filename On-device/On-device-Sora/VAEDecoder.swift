import Foundation
import CoreML

public struct VAEDecoder {
  
  var config: MLModelConfiguration
  
  public init(config: MLModelConfiguration) {
    self.config = config
  }

  func decode(latentVars: MLShapedArray<Float32>) async throws -> MLMultiArray {
      let startVAEDecodeTime = DispatchTime.now()

      print("Begin Decoding")
      let latentVarsTensor = MLTensor(latentVars).split(count: 4, alongAxis: 2)
      var temporalDecoded :[MLMultiArray] = []
      let spatialconfig = MLModelConfiguration()
      spatialconfig.computeUnits = .cpuAndGPU
      for i in 0...3 {
        print("========Temporal VAE\(i)=======")
        
        var temporalVAEPart1: vae_temporal_part1? = try vae_temporal_part1(configuration: self.config)
        let latentsTemporal = await latentVarsTensor[i].shapedArray(of: Float32.self)
        let inputTemporalVAEPart1 = vae_temporal_part1Input(latents: latentsTemporal)
        let temporalVAEPart1Output = try await temporalVAEPart1!.prediction(input: inputTemporalVAEPart1).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart1 = nil
        print("Part1 Decoding Complete")
        var temporalVAEPart2_1: vae_temporal_part2_1? = try vae_temporal_part2_1(configuration: self.config)
        let inputTemporalVAEPart2_1 = vae_temporal_part2_1Input(latents: temporalVAEPart1Output!)
        let temporalVAEPart2_1Output = try await temporalVAEPart2_1!.prediction(input: inputTemporalVAEPart2_1).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart2_1 = nil
        print("Part2-1 Decoding Complete")
        var temporalVAEPart2_2: vae_temporal_part2_2? = try vae_temporal_part2_2(configuration: spatialconfig)
        let inputTemporalVAEPart2_2 = vae_temporal_part2_2Input(latents: temporalVAEPart2_1Output!)
        let temporalVAEPart2_2Output = try await temporalVAEPart2_2!.prediction(input: inputTemporalVAEPart2_2).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart2_2 = nil
        print("Part2-2 Decoding Complete")
        var temporalVAEPart2_3_1: vae_temporal_part2_3_1? = try vae_temporal_part2_3_1(configuration: self.config)
        let inputTemporalVAEPart2_3_1 = vae_temporal_part2_3_1Input(latents: temporalVAEPart2_2Output!)
        let temporalVAEPart2_3_1Output = try await temporalVAEPart2_3_1!.prediction(input: inputTemporalVAEPart2_3_1).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart2_3_1 = nil
        
        var temporalVAEPart2_3_2: vae_temporal_part2_3_2? = try vae_temporal_part2_3_2(configuration: self.config)
        let inputTemporalVAEPart2_3_2 = vae_temporal_part2_3_2Input(latents: temporalVAEPart2_3_1Output!)
        let temporalVAEPart2_3_2Output = try await temporalVAEPart2_3_2!.prediction(input: inputTemporalVAEPart2_3_2).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart2_3_2 = nil
        
        var temporalVAEPart2_3_3: vae_temporal_part2_3_3? = try vae_temporal_part2_3_3(configuration: self.config)
        let inputTemporalVAEPart2_3_3 = vae_temporal_part2_3_3Input(latents: temporalVAEPart2_3_2Output!)
        let temporalVAEPart2_3_3Output = try await temporalVAEPart2_3_3!.prediction(input: inputTemporalVAEPart2_3_3).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart2_3_3 = nil
        
        var temporalVAEPart2_3_4: vae_temporal_part2_3_4? = try vae_temporal_part2_3_4(configuration: self.config)
        let inputTemporalVAEPart2_3_4 = vae_temporal_part2_3_4Input(latents: temporalVAEPart2_3_3Output!)
        let temporalVAEPart2_3_4Output = try await temporalVAEPart2_3_4!.prediction(input: inputTemporalVAEPart2_3_4).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart2_3_4 = nil
        
        print("Part2-3 Decoding Complete")
        var temporalVAEPart2_4: vae_temporal_part2_4? = try vae_temporal_part2_4(configuration: self.config)
        let inputTemporalVAEPart2_4 = vae_temporal_part2_4Input(latents: temporalVAEPart2_3_4Output!)
        let temporalVAEPart2_4Output = try await temporalVAEPart2_4!.prediction(input: inputTemporalVAEPart2_4).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart2_4 = nil
        print("Part2-4 Decoding Complete")
        var temporalVAEPart3: vae_temporal_part3? = try vae_temporal_part3(configuration: self.config)
        let inputTemporalVAEPart3 = vae_temporal_part3Input(latents: temporalVAEPart2_4Output!)
        let temporalVAEOutput = try await temporalVAEPart3!.prediction(input: inputTemporalVAEPart3).featureValue(for: "output")?.multiArrayValue
        temporalVAEPart3 = nil
        print("Part3 Decoding Complete")
        temporalDecoded.append(temporalVAEOutput!)
      }
    
      let decodedTemporalValues = MLMultiArray(concatenating: temporalDecoded, axis: 2, dataType: MLMultiArrayDataType.float32)
      
      let test = MLTensor(MLShapedArray<Float32>(decodedTemporalValues)).split(count: 17, alongAxis: 2)
    
      var decoded :[MLMultiArray] = []
    
      print("Spatial VAE Start")

      for i in 0...16 {
          print("========Spatial VAE\(i)=======")
        let latents = await test[i].shapedArray(of: Float32.self)
        let inputSpatial_1 = vae_spatial_part1Input(latents: latents)
        var spatialVAE_1: vae_spatial_part1? = try vae_spatial_part1(configuration: spatialconfig)
        print("Load Spatial VAE Part 1")
        let spatialVAEOutput_1 = try await spatialVAE_1!.prediction(input: inputSpatial_1).featureValue(for: "output")?.multiArrayValue
        spatialVAE_1 = nil
        
        let inputSpatial_2 = vae_spatial_part2Input(latents: spatialVAEOutput_1!)
        var spatialVAE_2:vae_spatial_part2? = try vae_spatial_part2(configuration: spatialconfig)
        print("Load Spatial VAE Part 2")
        let spatialVAEOutput_2 = try await spatialVAE_2!.prediction(input: inputSpatial_2).featureValue(for: "output")?.multiArrayValue
        spatialVAE_2 = nil
        
        let inputSpatial_3_1 = vae_spatial_part3_1Input(latents: spatialVAEOutput_2!)
        var spatialVAE_3_1: vae_spatial_part3_1? = try vae_spatial_part3_1(configuration: spatialconfig)
        print("Load Spatial VAE Part 3-1")
        let spatialVAEOutput_3_1 = try await spatialVAE_3_1!.prediction(input: inputSpatial_3_1).featureValue(for: "output")?.multiArrayValue
        spatialVAE_3_1 = nil
        
        let inputSpatial_3_2 = vae_spatial_part3_2Input(latents: spatialVAEOutput_3_1!)
        var spatialVAE_3_2: vae_spatial_part3_2? = try vae_spatial_part3_2(configuration: spatialconfig)
        print("Load Spatial VAE Part 3-2")
        let spatialVAEOutput_3_2 = try await spatialVAE_3_2!.prediction(input: inputSpatial_3_2).featureValue(for: "output")?.multiArrayValue
        spatialVAE_3_2 = nil
        
        let inputSpatial_3_3 = vae_spatial_part3_3Input(latents: spatialVAEOutput_3_2!)
        var spatialVAE_3_3: vae_spatial_part3_3? = try vae_spatial_part3_3(configuration: self.config)
        print("Load Spatial VAE Part 3-3")
        let spatialVAEOutput_3_3 = try await spatialVAE_3_3!.prediction(input: inputSpatial_3_3).featureValue(for: "output")?.multiArrayValue
        spatialVAE_3_3 = nil
        
        let inputSpatial_3_4_1 = vae_spatial_part3_4_1Input(latents: spatialVAEOutput_3_3!)
        var spatialVAE_3_4_1: vae_spatial_part3_4_1? = try vae_spatial_part3_4_1(configuration:  self.config)
        print("Load Spatial VAE Part 3-4")
        let spatialVAEOutput_3_4_1 = try await spatialVAE_3_4_1!.prediction(input: inputSpatial_3_4_1).featureValue(for: "output")?.multiArrayValue
        spatialVAE_3_4_1 = nil
        
        let inputSpatial_3_4_2 = vae_spatial_part3_4_2Input(latents: spatialVAEOutput_3_4_1!)
        var spatialVAE_3_4_2: vae_spatial_part3_4_2? = try vae_spatial_part3_4_2(configuration: self.config)
        print("Load Spatial VAE Part 3-4-2")
        let spatialVAEOutput_3_4_2 = try await spatialVAE_3_4_2!.prediction(input: inputSpatial_3_4_2).featureValue(for: "output")?.multiArrayValue
        spatialVAE_3_4_2 = nil
        
        let inputSpatial_3_4_3 = vae_spatial_part3_4_3Input(latents: spatialVAEOutput_3_4_2!)
        var spatialVAE_3_4_3: vae_spatial_part3_4_3? = try vae_spatial_part3_4_3(configuration: self.config)
        print("Load Spatial VAE Part 3-4-3")
        let spatialVAEOutput_3_4_3 = try await spatialVAE_3_4_3!.prediction(input: inputSpatial_3_4_3).featureValue(for: "output")?.multiArrayValue
        spatialVAE_3_4_3 = nil
        
        let inputSpatial_4 = vae_spatial_part4Input(latents: spatialVAEOutput_3_4_3!)
        var spatialVAE_4: vae_spatial_part4? = try vae_spatial_part4(configuration: spatialconfig)
        print("Load Spatial VAE Part 4")
        let spatialVAEOutput_4 = try await spatialVAE_4!.prediction(input: inputSpatial_4).featureValue(for: "output")?.multiArrayValue
        spatialVAE_4 = nil
        
        decoded.append(spatialVAEOutput_4!)
      }

      let decodedValues = MLMultiArray(concatenating: decoded, axis: 2, dataType: MLMultiArrayDataType.float32)

      print("Done Decoding")

      // Log time and return output
      let endVAEDecodeTime = DispatchTime.now()
      let elapsedVAEDecodeTime = endVAEDecodeTime.uptimeNanoseconds - startVAEDecodeTime.uptimeNanoseconds
      print("VAE Decode Running Time: \(Double(elapsedVAEDecodeTime) / 1000000000) seconds")
    return decodedValues
  }

  
    
    
}
