import SwiftUI

struct ContentView: View {
    @State var isGenerating: Bool = false
    
    @State var prompt: String = "a beautiful waterfall"
    @State private var seed = "42"
    @State private var aestheticScore = "6.5"
    @State private var step = 30
    @State private var mergeStep = 15
    @State private var numLpltarget = 15
    
    @State private var isBase = true
    @State private var isLPL = false
    @State private var isTDTM = false
    @State private var isCI = false
    @State private var isDL = false
  
    @StateObject private var tensor2vidConverter = Tensor2Vid()
  
    var body: some View {
        List {
          VStack(alignment: .leading) {
            Text("Prompt:")
            TextField("Enter prompt,but default exists", text: $prompt).padding(4).background(Color(uiColor: .secondarySystemBackground))
          }.listRowSeparator(.hidden)
          HStack {
            VStack(alignment: .leading) {
                        Text("LPL").font(.system(size: 22, weight: .semibold)).lineLimit(2)
                        HStack {
                            if self.isLPL {
                                Text("On")
                            } else {
                                Text("Off")
                            }
                            Spacer()
                            Toggle("", isOn: $isLPL)
                        }
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(self.isLPL ? Color.green: Color.gray, lineWidth: 2)
                    )
            VStack(alignment: .leading) {
                        Text("TDTM").font(.system(size: 22, weight: .semibold)).lineLimit(2)
                        HStack {
                          if self.isTDTM {
                                Text("On")
                            } else {
                                Text("Off")
                            }
                            Spacer()
                          Toggle("", isOn: $isTDTM)
                        }
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                          .stroke(self.isTDTM ? Color.green: Color.gray, lineWidth: 2)
                    )
          }.listRowSeparator(.hidden).frame(maxWidth: .infinity, alignment: .center)
          HStack{
            VStack(alignment: .leading) {
                        Text("CI").font(.system(size: 22, weight: .semibold)).lineLimit(2)
                        HStack {
                          if self.isCI {
                                Text("On")
                            } else {
                                Text("Off")
                            }
                            Spacer()
                            Toggle("", isOn: $isCI)
                        }
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(self.isCI ? Color.green: Color.gray, lineWidth: 2)
                    )
            VStack(alignment: .leading) {
                        Text("DL").font(.system(size: 22, weight: .semibold)).lineLimit(2)
                        HStack {
                            if self.isDL {
                                Text("On")
                            } else {
                                Text("Off")
                            }
                            Spacer()
                            Toggle("", isOn: $isDL)
                        }
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(self.isDL ? Color.green: Color.gray, lineWidth: 2)
                    )
          }.listRowSeparator(.hidden).frame(maxWidth: .infinity, alignment: .center)
          HStack(alignment: .center) {
            VStack(alignment: .leading) {
                      Text("192x192").font(.system(size: 22, weight: .semibold)).lineLimit(2).foregroundColor(self.isBase ? Color.gray : Color.green)
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                          .stroke(self.isBase ? Color.gray : Color.green, lineWidth: 2)
                    )
                    .onTapGesture {
                      self.isBase = false
                    }
            VStack(alignment: .leading) {
                      Text("256x256").font(.system(size: 22, weight: .semibold)).lineLimit(2).foregroundColor(self.isBase ? Color.green: Color.gray)
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                          .stroke(self.isBase ? Color.green: Color.gray, lineWidth: 2)
                    )
                    .onTapGesture {
                      self.isBase = true
                    }
          }.listRowSeparator(.hidden).frame(maxWidth: .infinity, alignment: .center)
          
          VStack(alignment: .leading) {
            Text("Seed:")
            TextField("42", text: $seed).keyboardType(.decimalPad).padding(4).background(Color(uiColor: .secondarySystemBackground))
          }.listRowSeparator(.hidden)
          
          VStack(alignment: .leading) {
            Text("Aesthetic score:")
            TextField("6.5", text: $aestheticScore).keyboardType(.decimalPad).padding(4).background(Color(uiColor: .secondarySystemBackground))
          }.listRowSeparator(.hidden)
          
          Stepper(
            value: $step,
            in: 0...50,
            step: 1
          ) {
            Text("Iteration steps: \(step)")
          }.listRowSeparator(.hidden)
          Stepper(
            value: $mergeStep,
            in: 0...step,
            step: 1
          ) {
            Text("Merge steps: \(mergeStep)")
          }.listRowSeparator(.hidden)
          
          Stepper(
            value: $numLpltarget,
            in: 0...50,
            step: 1
          ) {
            Text("LPL target steps: \(numLpltarget)")
          }.listRowSeparator(.hidden)
          
          if isGenerating {
            if let videoURL = tensor2vidConverter.videoURL {
                VideoPlayerView(url: videoURL).listRowSeparator(.hidden).frame(maxWidth: .infinity, alignment: .center)
            } else {
              ProgressView().padding().listRowSeparator(.hidden).frame(maxWidth: .infinity, alignment: .center)
            }
          }
          
          Button(action: generate) {
            Text("Start Video Generation").font(.title)
          }.buttonStyle(.borderedProminent).listRowSeparator(.hidden).frame(maxWidth: .infinity, alignment: .center)
        }
        .listStyle(.plain)
        .padding(.top)
        

    }
  
  func generate() {
      do {
        isGenerating = true
        let soraPipeline = try SoraPipeline(resourcesAt: Bundle.main.bundleURL, videoConverter: tensor2vidConverter)
        print("Start Video Generation")
        let aesprompt = prompt.appending(" aesthetic score: \(aestheticScore).")
        let logdir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        soraPipeline.sample(prompt: aesprompt, logdir: logdir, seed: Int(seed) ?? 42, step: step, mergeStep: mergeStep, numLpltarget: numLpltarget, isBase:isBase, isLPL: isLPL, isTDTM: isTDTM, isCI: isCI, isDL: isDL)
      } catch {
          print("Error: Can't initiallize SoraPipeline")
      }
    }
}
