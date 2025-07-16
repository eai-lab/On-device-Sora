import Accelerate
import CoreML

public final class RFlowScheduler {
  public let numTimesteps: Int
  
  public init(numTimesteps: Int) {
    self.numTimesteps = numTimesteps
  }
  
  public func addNoise(original_samples:MLTensor, noise: MLTensor, timesteps: Float32) -> MLTensor {
    var timepoints = MLTensor([1.0 - (Float32(timesteps) / Float32(self.numTimesteps))])
    
    timepoints = timepoints.expandingShape(at: 1).expandingShape(at: 1).expandingShape(at: 1).expandingShape(at: 1)
    timepoints = timepoints.tiled(multiples: [noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4]])
    return timepoints * original_samples + (1.0 - timepoints) * noise
    
  }
}
