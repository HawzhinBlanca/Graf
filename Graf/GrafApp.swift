import AppKit
import SwiftUI

@main
struct GrafApp: App {
    init() {
        print("GrafApp: Initializing application")
        NSLog("DEBUG: GrafApp initializing")
    }

    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear {
                    print("GrafApp: ContentView appeared")
                    NSLog("DEBUG: GrafApp ContentView appeared")
                }
                .frame(minWidth: 800, minHeight: 600)
        }
    }
}
