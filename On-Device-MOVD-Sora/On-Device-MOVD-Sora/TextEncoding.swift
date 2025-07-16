import Foundation
import Tokenizers
import CoreML

public struct TextEncoderT5Output {
  public let encoderHiddenStates: MLShapedArray<Float32>
  public let masks: MLShapedArray<Float32>
  public let yNull: MLShapedArray<Float32>
}

public struct TextEncoding {
  var tokenizer: Tokenizer
  var DivT5s: [ManagedMLModel]
  var embed: ManagedMLModel
  var finalNorm: ManagedMLModel
  
  init(tokenizer: Tokenizer, DivT5s: [ManagedMLModel], embed: ManagedMLModel, finalNorm: ManagedMLModel) {
    self.tokenizer = tokenizer
    self.DivT5s = DivT5s
    self.embed = embed
    self.finalNorm = finalNorm
  }
  
  func tokenize(_ text: String) throws -> [Int] {
    // Get models expected input length
    let inputLength = inputShape.last!
    // Tokenize, padding to the expected length
    var tokens = tokenizer.tokenize(text: text)
    var ids = tokens.map { tokenizer.convertTokenToId($0) ?? 0 }
    // Truncate if necessary
    if ids.count > inputLength {
        tokens = tokens.dropLast(tokens.count - inputLength)
        ids = ids.dropLast(ids.count - inputLength)
        print("Needed to truncate input for TextEncoderT5")
    }
    print("Done tokenizing")
    return ids
  }
  
  func encode(ids: [Int]) throws -> TextEncoderT5Output {
    let startT5Time = DispatchTime.now()

    let inputName = "input_ids"
    let inputShape = [1,300,4096]
    let inputLength = inputShape[1]
            
    let bosToken = tokenizer.bosTokenId ?? 0
    let eosToken = tokenizer.eosTokenId ?? 1
    let padToken = bosToken
    let maskToken = -Float32.greatestFiniteMagnitude
    let truncatedIds = ids.prefix(inputLength - 1) + [eosToken]
    print("Result of Tokenizing: \(truncatedIds)")
    let inputIds = truncatedIds + Array(repeating: padToken, count: inputLength - truncatedIds.count)

    var attentionMask: [Float32] = inputIds.map { token in
      token == padToken ? maskToken : 0.0
    }
    attentionMask[0] = 0.0

    let floatIds = inputIds.map { Float32($0) }

    let inputShapeEmbed = inputShapeEmbed
    let inputArrayEmbed =  MLShapedArray<Float32>(scalars: floatIds, shape: inputShapeEmbed)
    let inputFeaturesEmbed = try! MLDictionaryFeatureProvider(dictionary: [inputName: MLMultiArray(inputArrayEmbed)])

    
    let resultEmbed = try embed.perform { model in
      try model.prediction(from: inputFeaturesEmbed)
    }
    embed.unloadResources()

    print("Done Embedding")
    
    let maskArray = MLShapedArray<Float32>(scalars: attentionMask, shape: [1,1,1,300])
    var inputFeatures: MLFeatureProvider = try! MLDictionaryFeatureProvider(
      dictionary: ["hidden_states": resultEmbed.featureValue(for: "output") as Any,
                   "attention_mask": MLMultiArray(maskArray),
                   "y_embedding": MLMultiArray(MLShapedArray(repeating: 1.0, shape: [300,4096]))
                  ])

    print("Processing DivT5s")
    var position_bias: Any? = nil
    var yNull: MLMultiArray? = nil
    
    let t5loadQueue = DispatchQueue(label: "T5LoadQueue", qos: .background, attributes: .concurrent)
    let t5computeQueue = DispatchQueue(label: "T5ComputeQueue", qos: .userInitiated)
    for (index,model) in DivT5s.enumerated() {
        let t5group = DispatchGroup()
        t5group.enter()
        t5computeQueue.async {
            do {
                let layerOutputs = try model.perform { model in
                  try model.prediction(from: inputFeatures)
                }
                model.unloadResources()
                if index == 0 {
                  position_bias = layerOutputs.featureValue(for: "output_position_bias")
                  yNull = layerOutputs.featureValue(for: "yNull")?.multiArrayValue
                }
                inputFeatures = try MLDictionaryFeatureProvider(
                  dictionary: ["hidden_states": layerOutputs.featureValue(for: "output_hidden_states") as Any,
                               "attention_mask": MLMultiArray(maskArray),
                               "position_bias" : position_bias as Any])
                  print("Done T5_layer_\(index)_Block")
                } catch {
                    print("Falied to perform T5 prediction")
                }
                t5group.leave()
        }
        if index < DivT5s.count - 1 {
            let nextModel = DivT5s[index + 1]
            //t5group.enter()
            t5loadQueue.async {
                do {
                    try nextModel.loadResources()
                } catch {
                    print("Failed to load next T5 Model")
                }
                //t5group.leave()
            }
        }
        t5group.wait()
    }
      
    print("Done DivT5s")
    let hidden_states = inputFeatures.featureValue(for: "hidden_states")

    let inputFeaturesNorm = try! MLDictionaryFeatureProvider(dictionary: ["input": hidden_states as Any])

    let resultNorm = try finalNorm.perform { model in
      try model.prediction(from: inputFeaturesNorm)
    }
    finalNorm.unloadResources()
    print("Done Final Norm")
      
    print("Done T5 processing")
    
    // For Debugging
    let endT5Time = DispatchTime.now()
    let elapsedT5Time = endT5Time.uptimeNanoseconds - startT5Time.uptimeNanoseconds
    print("T5 Running Time: \(Double(elapsedT5Time) / 1000000000)")
    
    let returnAttentionMask: [Float32] = inputIds.map { token in
      token == padToken ? 0.0 : 1.0
    }
    let returnAttentionMaskArray = MLShapedArray<Float32>(scalars: returnAttentionMask, shape: [1,1,1,300])

    return TextEncoderT5Output(encoderHiddenStates: MLShapedArray<Float32>(converting: resultNorm.featureValue(for: "output")!.multiArrayValue!).expandingShape(at: 0), masks: returnAttentionMaskArray.squeezingShape().expandingShape(at: 0), yNull: MLShapedArray<Float32>(converting: yNull!).expandingShape(at: 0).expandingShape(at: 0))
  }
  
  var inputDescription: MLFeatureDescription {
      try! DivT5s[0].perform { model in
          model.modelDescription.inputDescriptionsByName.first!.value
      }
  }
  
  var inputDescriptionEmbed: MLFeatureDescription {
    try! embed.perform { model in
        model.modelDescription.inputDescriptionsByName.first!.value
    }
  }
  
  var inputShape: [Int] {
      inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
  }
  
  var inputShapeEmbed: [Int] {
    inputDescriptionEmbed.multiArrayConstraint!.shape.map { $0.intValue}
  }
}
