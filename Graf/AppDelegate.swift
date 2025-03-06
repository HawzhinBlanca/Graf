import Cocoa
import SwiftUI
import os.log

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    private let logger = Logger(subsystem: "com.hawzhin.Graf", category: "AppDelegate")

    // Error handling
    private var startupError: Error? = nil

    func applicationDidFinishLaunching(_ notification: Notification) {
        logger.info("Application did finish launching")

        do {
            // Create the SwiftUI view that provides the window contents.
            let contentView = ContentView()
            logger.debug("ContentView created")

            // Create the window and set the content view
            window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
                styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                backing: .buffered,
                defer: false
            )
            logger.debug("NSWindow created")

            window.title = "Graf - 4D Geometry Visualizer"
            window.center()
            window.setFrameAutosaveName("Main Window")
            window.contentView = NSHostingView(rootView: contentView)
            logger.debug("ContentView set to window")

            window.makeKeyAndOrderFront(nil)

            // Set minimum window size for better UI experience
            window.minSize = NSSize(width: 800, height: 600)

            // Force the window to be visible
            NSApp.activate(ignoringOtherApps: true)
            logger.info("Window initialization completed successfully")

            // Check for updates in the background
            checkForUpdatesInBackground()

        } catch {
            startupError = error
            logger.error("Failed to initialize application: \(error.localizedDescription)")
            showStartupErrorAlert(error: error)
        }
    }

    private func showStartupErrorAlert(error: Error) {
        let alert = NSAlert()
        alert.messageText = "Application Error"
        alert.informativeText = "Could not start Graf: \(error.localizedDescription)"
        alert.alertStyle = .critical
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    private func checkForUpdatesInBackground() {
        // This would be implemented with your preferred update mechanism
        // For example, Sparkle framework or a custom implementation
        logger.debug("Checking for updates in background")

        // Placeholder for update check logic
        DispatchQueue.global(qos: .background).async {
            // Check for updates logic would go here
            self.logger.debug("Update check completed")
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        logger.info("Application will terminate")

        // Save any unsaved preferences
        UserDefaults.standard.synchronize()

        // Clean up resources if needed
        // ...
    }

    // Handle file open events
    func application(_ application: NSApplication, open urls: [URL]) {
        for url in urls {
            logger.info("Opening file: \(url.lastPathComponent)")
            // Handle file opening logic here
        }
    }
}
