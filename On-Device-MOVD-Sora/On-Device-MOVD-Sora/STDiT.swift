import CoreML

public struct STDiT {
  var part1: ManagedMLModel
  var spatialAndTemporals: [ManagedMLModel]
  var STDiTTDTMBLocks: [ManagedMLModel]
  var part2: ManagedMLModel
  var memoryInfo: MemoryInfo
  
  init(part1: ManagedMLModel, spatialAndTemporals: [ManagedMLModel], STDiTTDTMBLocks: [ManagedMLModel], part2: ManagedMLModel) {
    self.part1 = part1
    self.spatialAndTemporals = spatialAndTemporals
    self.STDiTTDTMBLocks = STDiTTDTMBLocks
    self.part2 = part2
    self.memoryInfo = MemoryInfo(beforeMemory: 0, afterMemory: 0, needMemory: 0, loadMemory: 0, remainMemory: 0, countOfUnload: 0)
  }
  
  func sample(x:MLShapedArray<Float32>, timestep: MLShapedArray<Float32>, modelargs:Dictionary<String, MLShapedArray<Float32>>, step: Int, numSamplingsteps: Int, mergeStep: Int, numLpltarget: Int, isLPL: Bool = false, isTDTM: Bool = false, isCI: Bool = false, isDL: Bool = false ) async throws -> MLTensor {
    // === Start layer ===
    var inputFeatures = try MLDictionaryFeatureProvider(
      dictionary: ["z_in": MLMultiArray(x), "t": MLMultiArray(timestep), "y": MLMultiArray(modelargs["y"]!), "mask": MLMultiArray(modelargs["mask"]!), "fps": MLMultiArray(modelargs["fps"]!), "height": MLMultiArray(modelargs["height"]!), "width": MLMultiArray(modelargs["width"]!), "padH": MLMultiArray(modelargs["padH"]!), "padW": MLMultiArray(modelargs["padW"]!)]
    )
    let stdit3Part1Output = try part1.perform { model in
      try model.prediction(from: inputFeatures)
    }
    part1.unloadResources()
    
    print("=== Done stdit3 Part1 ===")
    
    inputFeatures = try MLDictionaryFeatureProvider (
      dictionary : [ "x" : stdit3Part1Output.featureValue(for: "x")!, "y" : stdit3Part1Output.featureValue(for: "outY")!, "attn": MLMultiArray(modelargs["BDM"]!), "t_mlp": stdit3Part1Output.featureValue(for: "t_mlp")!, "T": stdit3Part1Output.featureValue(for: "T")!])

    
    
    // === blocks ===
    let loadQueue = DispatchQueue(label: "modelLoadQueue", qos: .background, attributes: .concurrent)
    let computeQueue = DispatchQueue(label: "modelComputeQueue", qos: .userInitiated)
    for (i, spatialAndTemporal) in spatialAndTemporals.enumerated() {
      print("==== Block \(i) ====")
      let startTotalTime = DispatchTime.now()
      
      let group = DispatchGroup()
      
      group.enter()
      computeQueue.async { [self] in
        do {
          if i==0 && step==0 && isDL {
            memoryInfo.remainMemory = os_proc_available_memory()
          }
          try spatialAndTemporal.loadResources()

          if i==0 && step == mergeStep && isDL {
            memoryInfo.remainMemory = os_proc_available_memory()
          }
          memoryInfo.loadMemory = 170*1000*1000
          let spatialOutput = try spatialAndTemporal.loadedModel?.prediction(from: inputFeatures)
          if i==0 && step==0 && isDL {
            memoryInfo.needMemory = Int(getMemoryUsedAndDeviceTotalInMegabytes())
          }
          if i==0 && step == mergeStep && isDL {
            memoryInfo.needMemory = Int(getMemoryUsedAndDeviceTotalInMegabytes())
          }
          if memoryInfo.remainMemory - memoryInfo.loadMemory > memoryInfo.needMemory + memoryInfo.loadMemory * 2  && isDL {
            memoryInfo.remainMemory -= memoryInfo.loadMemory
            memoryInfo.countOfUnload += 1
          } else {
            if i >= memoryInfo.countOfUnload || step == numSamplingsteps - 1 || step == numLpltarget {
              spatialAndTemporal.unloadResources()
            }
          }
          
          inputFeatures = try MLDictionaryFeatureProvider (
            dictionary : [ "x" : spatialOutput!.featureValue(for: "output")!, "y" : stdit3Part1Output.featureValue(for: "outY")!, "attn": MLMultiArray(modelargs["BDM"]!), "t_mlp": stdit3Part1Output.featureValue(for: "t_mlp")!, "T": stdit3Part1Output.featureValue(for: "T")!])
                         
          
        } catch {
          print("Failed to perform prediction")
        }
        group.leave()
      }
      
      if isCI && step > 0 {
        if i < spatialAndTemporals.count - 1 {
          if step == 0 || i >= memoryInfo.countOfUnload - 1 {
            let nextModel = spatialAndTemporals[i + 1]
            loadQueue.async {
              do {
                try nextModel.loadResources()
              } catch {
                print("Failed to load next Model")
              }
            }
          }
        }
      }
      group.wait()
      let endTotalTime = DispatchTime.now()
      let elapsedTotalTime = endTotalTime.uptimeNanoseconds - startTotalTime.uptimeNanoseconds
      print("STDiT_\(i) Running Time: \(Double(elapsedTotalTime) / 1000000000) seconds")
    }
    
    // === final layer ===
    inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["z_in": MLMultiArray(x), "x": (inputFeatures.featureValue(for: "x")?.multiArrayValue!)!, "t": (stdit3Part1Output.featureValue(for: "outT")?.multiArrayValue!)!, "padH": MLMultiArray(modelargs["padH"]!), "padW": MLMultiArray(modelargs["padW"]!)])

    
    let stdit3Part2Output = try part2.perform { model in
      try model.prediction(from: inputFeatures)
    }
    part2.unloadResources()
    
    let output = MLTensor(MLShapedArray<Float32>((stdit3Part2Output.featureValue(for: "output")?.multiArrayValue)!))
    return output
  }
  
  func sampleTDTM(x:MLShapedArray<Float32>, timestep: MLShapedArray<Float32>, modelargs:Dictionary<String, MLShapedArray<Float32>>, step: Int, numSamplingsteps: Int, mergeStep: Int, numLpltarget: Int, isLPL: Bool = false, isTDTM: Bool = false, isCI: Bool = false, isDL: Bool = false ) async throws -> MLTensor {
    print("Sampling with TDTM")
    // === Start layer ===
    var inputFeatures = try MLDictionaryFeatureProvider(
      dictionary: ["z_in": MLMultiArray(x), "t": MLMultiArray(timestep), "y": MLMultiArray(modelargs["y"]!), "mask": MLMultiArray(modelargs["mask"]!), "fps": MLMultiArray(modelargs["fps"]!), "height": MLMultiArray(modelargs["height"]!), "width": MLMultiArray(modelargs["width"]!), "padH": MLMultiArray(modelargs["padH"]!), "padW": MLMultiArray(modelargs["padW"]!)]
    )
    let stdit3Part1Output = try part1.perform { model in
      try model.prediction(from: inputFeatures)
    }
    part1.unloadResources()
    
    print("=== Done stdit3 Part1 ===")
    
    inputFeatures = try MLDictionaryFeatureProvider (
      dictionary : [ "x" : stdit3Part1Output.featureValue(for: "x")!, "y" : stdit3Part1Output.featureValue(for: "outY")!, "attn": MLMultiArray(modelargs["BDM"]!), "t_mlp": stdit3Part1Output.featureValue(for: "t_mlp")!, "T": stdit3Part1Output.featureValue(for: "T")!])
    
    
    // === blocks ===
    let loadTDTMQueue = DispatchQueue(label: "modelLoadTDTMQueue", qos: .background, attributes: .concurrent)
    let computeQueue = DispatchQueue(label: "modelComputeQueue", qos: .userInitiated)
    for (i, spatialAndTemporal) in STDiTTDTMBLocks.enumerated() {
      print("==== Block \(i) ====")
      let startTotalTime = DispatchTime.now()

      let group = DispatchGroup()
      group.enter()
      
      computeQueue.async {
        do {
          if i==0 && step==0 && isDL {
            memoryInfo.remainMemory = os_proc_available_memory()
          }
          try spatialAndTemporal.loadResources()
          memoryInfo.loadMemory = 170*1000*1000
          let spatialOutput = try spatialAndTemporal.loadedModel?.prediction(from: inputFeatures)
          if i==0 && step==0 && isDL {
            memoryInfo.needMemory = Int(getMemoryUsedAndDeviceTotalInMegabytes())
          }
          if memoryInfo.remainMemory - memoryInfo.loadMemory > memoryInfo.needMemory + memoryInfo.loadMemory * 2 && isDL {
            memoryInfo.remainMemory -= memoryInfo.loadMemory
            memoryInfo.countOfUnload += 1

          } else {
            if i >= memoryInfo.countOfUnload || step == mergeStep - 1 || step == numLpltarget {
              spatialAndTemporal.unloadResources()
            }
          }
          
          inputFeatures = try MLDictionaryFeatureProvider (
            dictionary : [ "x" : spatialOutput!.featureValue(for: "output")!, "y" : stdit3Part1Output.featureValue(for: "outY")!, "attn": MLMultiArray(modelargs["BDM"]!), "t_mlp": stdit3Part1Output.featureValue(for: "t_mlp")!, "T": stdit3Part1Output.featureValue(for: "T")!])
          
        } catch {
          print("Failed to perform prediction")
        }
        group.leave()
      }
      
      if isCI && step > 0 {
        if i < STDiTTDTMBLocks.count - 1 {
          if step == 0 || i >= memoryInfo.countOfUnload - 1 {
            let nextModel = STDiTTDTMBLocks[i + 1]
            loadTDTMQueue.async {
              do {
                try nextModel.loadResources()
              } catch {
                print("Failed to load next Model")
              }
            }
          }
        }
      }
      group.wait()
      let endTotalTime = DispatchTime.now()
      let elapsedTotalTime = endTotalTime.uptimeNanoseconds - startTotalTime.uptimeNanoseconds
      print("STDiT_TDTM\(i) Running Time: \(Double(elapsedTotalTime) / 1000000000) seconds")
      
    }
    
    // === final layer ===
    inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["z_in": MLMultiArray(x), "x": (inputFeatures.featureValue(for: "x")?.multiArrayValue!)!, "t": (stdit3Part1Output.featureValue(for: "outT")?.multiArrayValue!)!, "padH": MLMultiArray(modelargs["padH"]!), "padW": MLMultiArray(modelargs["padW"]!)])
    
    let stdit3Part2Output = try part2.perform { model in
      try model.prediction(from: inputFeatures)
    }
    part2.unloadResources()
    
    let output = MLTensor(MLShapedArray<Float32>((stdit3Part2Output.featureValue(for: "output")?.multiArrayValue)!))
    return output
  }
}

extension STDiT {
  func getMemoryUsedAndDeviceTotalInMegabytes() -> UInt64 {
      
    var taskInfo = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info>.size) / 4
    let result: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    var used: UInt64 = 0
    if result == KERN_SUCCESS {
        used = UInt64(taskInfo.phys_footprint)
    }
    return used
  }
}
