import SwiftUI
import AVKit

struct VideoPlayerView: View {
    let url: URL

    var body: some View {
        VideoPlayer(player: AVPlayer(url: url))
            .aspectRatio(contentMode: .fit)
            .onAppear {
                // run the video player
                AVPlayer(url: url).play()
            }
    }
}
