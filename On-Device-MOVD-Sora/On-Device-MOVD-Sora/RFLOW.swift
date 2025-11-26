import Accelerate
import CoreML
import Foundation


public struct RFLOWInput {
  let model: STDiT
  let modelArgs: Dictionary<String, MLShapedArray<Float32>>
  let z: MLShapedArray<Float32>
  let mask: MLShapedArray<Float32>
  let additionalArgs: [String: MLTensor]
  let resolution: Double

}

public final class RFLOW {
  
  public let numSamplingsteps: Int
  public let numTimesteps: Int
  public let cfgScale: Float
  public let scheduler: RFlowScheduler
  public let seed: UInt64
  
  public let isLPL: Bool
  public let isTDTM: Bool
  public let isCI: Bool
  public let isDL: Bool
  
  // For TDTM ka
  public let mergeStep: Int
  public let idsCount: Int
  public let attnSize: Int
  
  // For LPL
  public var numLpltarget : Int
  public var best_cos : Float = 0.0
  public let cos_tolerance : Float = 0.0001
  public var tolerance_n : Int = 5
  
  
  public init(numSamplingsteps: Int = 10, numTimesteps: Int = 1000, cfgScale: Float = 4.0, seed: UInt64 = 42, idsCount: Int, attnSize: Int, mergeStep: Int = 0, numLpltarget: Int = 20, isLPL: Bool = false, isTDTM: Bool = false, isCI: Bool = false, isDL: Bool = false) {
    self.numSamplingsteps = numSamplingsteps
    self.numTimesteps = numTimesteps
    self.numLpltarget = numLpltarget
    self.cfgScale = cfgScale
    self.scheduler = RFlowScheduler(numTimesteps: numTimesteps)
    self.idsCount = idsCount
    self.attnSize = attnSize
    self.seed = seed
    self.mergeStep = mergeStep
    self.isLPL = isLPL
    self.isTDTM = isTDTM
    self.isCI = isCI
    self.isDL = isDL
  }
  
  public func sample(rflowInput: RFLOWInput, yNull: MLShapedArray<Float32>) async -> MLTensor {
    let guidanceScale = self.cfgScale
    
    // text encoding
    var modelArgs = rflowInput.modelArgs
    modelArgs["y"] = MLShapedArray(concatenating: [modelArgs["y"]!, yNull], alongAxis: 0)
    for (key, value) in rflowInput.additionalArgs {
      modelArgs[key] = await value.shapedArray(of: Float32.self)
    }
    // prepare timesteps
    var timeSteps: [Float32] = []
    for i in 0..<self.numSamplingsteps {
      var t = (1.0 - Float32(i) / Float32(self.numSamplingsteps)) * Float32(self.numTimesteps)
      t = timestep_transform(t: round(t), num_timesteps: self.numTimesteps, resolution: rflowInput.resolution)
      timeSteps.append(t)
    }
    
    let mask = MLTensor(rflowInput.mask)
    var noiseAdded = MLTensor(repeating: false, shape: mask.shape, scalarType: Bool.self)
    
    noiseAdded = noiseAdded .| (mask .== 1)
    let numTimestepsTensor = MLTensor(repeating: Float32(self.numTimesteps), shape: mask.shape)
    var z = MLTensor(rflowInput.z)
    let startTime = DispatchTime.now()
    var prev_velocity: MLTensor = MLTensor(repeating: 0.0, shape: rflowInput.z.shape, scalarType: Float32.self)

    for (i,t) in timeSteps.enumerated() {
      print("== Step \(i) ==")
      // mask for adding noise
      let mask_t = mask * numTimestepsTensor
      let x0 = z
      let xNoise = self.scheduler.addNoise(original_samples: x0, noise: MLTensor(randomNormal: x0.shape, seed: UInt64(seed),scalarType: Float32.self), timesteps: t)
      
      let T = MLTensor([Float32(t)])
      let maskTUpper = mask_t .>= T.expandingShape(at: 1)
      modelArgs["x_mask"] = await MLTensor(concatenating: [maskTUpper, maskTUpper], alongAxis: 0).cast(to: Float32.self).shapedArray(of: Float32.self)

      let maskAddNoise: MLTensor? = maskTUpper .& .!noiseAdded
      let expandedMaskAN = maskAddNoise!.cast(to: Float32.self).expandingShape(at: 1).expandingShape(at: 3).expandingShape(at: 4)
      z = expandedMaskAN * xNoise + (1.0 - expandedMaskAN) * x0
      
      noiseAdded = maskTUpper
      
      // classifier-free guidance
      let zIn = await MLTensor(concatenating: [z,z], alongAxis: 0).shapedArray(of: Float32.self)
      let tIn = await MLTensor(concatenating: [T,T], alongAxis: 0).shapedArray(of: Float32.self)
      var pred: MLTensor
      if isTDTM && i < mergeStep {
        modelArgs["BDM"] = makeAttnMask(count: idsCount, attnSize: attnSize/2)
      } else {
        modelArgs["BDM"] = makeAttnMask(count: idsCount, attnSize: attnSize)
      }

      if isTDTM && i < mergeStep {
        pred = try! await rflowInput.model.sampleTDTM(x: zIn, timestep: tIn, modelargs: modelArgs, step: i, numSamplingsteps: numSamplingsteps,mergeStep: mergeStep, numLpltarget: numLpltarget, isLPL: isLPL, isTDTM: isTDTM, isCI: isCI, isDL: isDL)
      } else {
        pred = try! await rflowInput.model.sample(x: zIn, timestep: tIn, modelargs: modelArgs, step: i, numSamplingsteps: numSamplingsteps, mergeStep: mergeStep, numLpltarget: numLpltarget, isLPL: isLPL, isTDTM: isTDTM, isCI: isCI, isDL: isDL)
      }
      
      if isLPL {
        //if sum of previous velocity is 0 == None, prev_vel is pred
        if await prev_velocity.sum().shapedArray(of: Float32.self) == MLShapedArray(arrayLiteral: 0.0) {
          prev_velocity = pred
        }
        
//        var shoot_flag = await self.compute_cos_sim(prev: prev_velocity, v_pred: pred, i: i)
//        
//        if shoot_flag == true {
//          self.numLpltarget = i
//        }
      }


      let splitSize1 = pred.shape[1] / 2
      pred = pred.split(sizes: [splitSize1,splitSize1], alongAxis: 1)[0] //chuck

      let splitSize2 = pred.shape[0] / 2
      let finalPred = pred.split(sizes: [splitSize2, splitSize2], alongAxis: 0) // chuck
      let vPred = finalPred[1] + guidanceScale * (finalPred[0] - finalPred[1])
      
      // update z
      var dt = i < timeSteps.count - 1 ? timeSteps[i] - timeSteps[i + 1] : timeSteps[i]
      
      if i == self.numLpltarget && isLPL {
        dt = timeSteps[i]
      }
      
      dt = Float32(dt) / Float32(self.numTimesteps)
      let DT = MLTensor([dt])
      
      z = z + vPred * DT.expandingShape(at: 1).expandingShape(at: 2).expandingShape(at: 3).expandingShape(at: 4)
      
      let expandedMaskTU = maskTUpper.cast(to: Float32.self).expandingShape(at: 1).expandingShape(at: 3).expandingShape(at: 4)
      z = expandedMaskTU * z + (1.0 - expandedMaskTU) * x0
      
      if i == self.numLpltarget && isLPL{
        // break the loop
        let endTime = DispatchTime.now()
        let elapsedTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
        print("STDit Running Time: \(Double(elapsedTime) / 1000000000)")
        return z
      }
    }
    
    
    let endTime = DispatchTime.now()
    let elapsedTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
    print("STDit Running Time: \(Double(elapsedTime) / 1000000000)")
    return z
  }
}

extension RFLOW {
  func makeAttnMask(count: Int, attnSize: Int)  -> MLShapedArray<Float32> {
          let blockdigonalmask = MLShapedArray(repeating: -Float32.greatestFiniteMagnitude, shape: [attnSize, count])
          let blockdigonalnonmask = MLShapedArray(repeating: Float32(0.0), shape: [attnSize, count])
          let blockMask = MLShapedArray(concatenating: [blockdigonalnonmask, blockdigonalmask], alongAxis: 1)
          let blockNonMask = MLShapedArray(concatenating: [blockdigonalmask, blockdigonalnonmask], alongAxis: 1)
          let blockConcat = MLShapedArray(concatenating: [blockMask, blockNonMask], alongAxis: 0).expandingShape(at: 0).expandingShape(at: 0)
          var concatList: [MLShapedArray<Float32>] = [] // for expand
          for _ in 0...15 {
          concatList.append(blockConcat)
      }
      return MLShapedArray(concatenating: concatList, alongAxis: 1)
  }
  
  func timestep_transform(t: Float32, num_timesteps: Int = 1, resolution: Double) -> Float32 {
    let base_resolution = 512.0 * 512.0
    
    let T = t / Float32(num_timesteps)
    let ratio_space = Float32(resolution / base_resolution).squareRoot()
    let ratio_time = Float32(Int(51 / 17) * 5).squareRoot()
    let ratio = ratio_space * ratio_time
    var new_t = ratio * T / (1 + (ratio - 1) * T)
    new_t = new_t * Float32(num_timesteps)
    return new_t
  }
  
  func compute_cos_sim(prev: MLTensor, v_pred: MLTensor,i:Int) async -> Bool{
      var shoot_flag = false
      if await prev.sum().shapedArray(of: Float32.self) != MLShapedArray(arrayLiteral: 0.0) {
        
        let prev_shape = prev.shape
        var pred_v_val = prev.split(count: prev_shape[0], alongAxis: 0)[0].split(count: prev_shape[1], alongAxis: 1)[0].split(count: prev_shape[2], alongAxis: 2)[0]
        var v_pred_val = v_pred.split(count: prev_shape[0], alongAxis: 0)[0].split(count: prev_shape[1], alongAxis: 1)[0].split(count: prev_shape[2], alongAxis: 2)[0]
        let pred_norm = pred_v_val.squareRoot().sum()
        let prev_norm = v_pred_val.squareRoot().sum()
        
        pred_v_val = pred_v_val / pred_norm
        v_pred_val = v_pred_val / prev_norm
        
        let cos_sim = pred_v_val.matmul(v_pred_val).sum()
        let cos_sim_mean = Float32(MLMultiArray(await cos_sim.mean().shapedArray(of: Float32.self))[0])
        if cos_sim_mean > self.best_cos + self.cos_tolerance {
          self.best_cos = cos_sim_mean
          self.tolerance_n = 5
          shoot_flag = false
        }
        
        else {
          self.tolerance_n -= 1
        }
        
        if tolerance_n == 0 {
          shoot_flag = true
        }
      }
      return shoot_flag
    }
}
