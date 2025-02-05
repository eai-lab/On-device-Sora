import CoreML

public final class ManagedMLModel {
  var modelURL: URL
  
  var config: MLModelConfiguration
  
  var loadedModel: MLModel?
  
  var queue: DispatchQueue
  
  public init(modelURL: URL, config: MLModelConfiguration) {
    self.modelURL = modelURL
    self.config = config
    self.loadedModel = nil
    self.queue = DispatchQueue(label: "managed.\(modelURL.lastPathComponent)")
  }
  
  public func loadResources() throws {
    try queue.sync {
      try loadModel()
    }
  }
  
  public func unloadResources() {
      queue.sync {
          loadedModel = nil
      }
  }

  public func perform<R>(_ body: (MLModel) throws -> R) throws -> R {
      return try queue.sync {
          try autoreleasepool {
              try loadModel()
              return try body(loadedModel!)
          }
      }
  }
  
  private func loadModel() throws {
    if loadedModel == nil {
      loadedModel = try MLModel(contentsOf: modelURL, configuration: config)
    }
  }
}
