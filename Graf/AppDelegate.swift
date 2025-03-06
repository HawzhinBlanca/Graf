import Cocoa
import SwiftUI

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSLog("DEBUG: Graf applicationDidFinishLaunching called")

        // Create the SwiftUI view that provides the window contents.
        let contentView = ContentView()
        NSLog("DEBUG: Graf ContentView created")

        // Create the window and set the content view
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        NSLog("DEBUG: Graf NSWindow created")

        window.title = "Graf - 4D Geometry Visualizer"
        window.center()
        window.setFrameAutosaveName("Main Window")
        window.contentView = NSHostingView(rootView: contentView)
        NSLog("DEBUG: Graf ContentView set to window")

        window.makeKeyAndOrderFront(nil)
        NSLog("DEBUG: Graf makeKeyAndOrderFront called")

        // Set minimum window size for better UI experience
        window.minSize = NSSize(width: 800, height: 600)

        // Force the window to be visible
        NSApp.activate(ignoringOtherApps: true)
        NSLog("DEBUG: Graf NSApp.activate called")
    }

    func applicationWillTerminate(_ notification: Notification) {
        NSLog("DEBUG: Graf applicationWillTerminate called")
        // Clean up resources if needed
    }
}
