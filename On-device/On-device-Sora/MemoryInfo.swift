public class MemoryInfo {
  var beforeMemory: Int
  var afterMemory: Int
  var needMemory: Int
  var loadMemory: Int
  var remainMemory: Int
  var countOfUnload: Int
  
  init(beforeMemory: Int, afterMemory: Int, needMemory: Int, loadMemory: Int, remainMemory: Int, countOfUnload: Int) {
    self.beforeMemory = beforeMemory
    self.afterMemory = afterMemory
    self.needMemory = needMemory
    self.loadMemory = loadMemory
    self.remainMemory = remainMemory
    self.countOfUnload = countOfUnload
  }
}
