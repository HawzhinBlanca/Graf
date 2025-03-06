import AppKit
import SwiftUI
import os.log

@main
struct GrafApp: App {
    // Create a logger for the app
    private let logger = Logger(subsystem: "com.hawzhin.Graf", category: "Application")

    // App-level state
    @State private var hasInitializationError: Bool = false
    @State private var initializationErrorMessage: String = ""

    init() {
        logger.info("GrafApp: Initializing application")

        // Setup app-specific configurations
        setupAppConfiguration()
    }

    private func setupAppConfiguration() {
        // Set default UserDefaults if not already set
        if UserDefaults.standard.object(forKey: "hasLaunchedBefore") == nil {
            UserDefaults.standard.set(true, forKey: "hasLaunchedBefore")
            UserDefaults.standard.set(true, forKey: "showWireframe")
            UserDefaults.standard.set(32, forKey: "defaultResolution")
            logger.info("First launch detected, setting default preferences")
        }

        #if DEBUG
            logger.debug("Running in DEBUG mode")
        #endif
    }

    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            if hasInitializationError {
                // Show error view
                VStack(spacing: 20) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.red)

                    Text("Initialization Error")
                        .font(.title)
                        .fontWeight(.bold)

                    Text(initializationErrorMessage)
                        .multilineTextAlignment(.center)
                        .padding()

                    Button("Try Again") {
                        // Attempt to reload
                        hasInitializationError = false
                    }
                    .padding()
                    .buttonStyle(.borderedProminent)
                }
                .frame(minWidth: 400, minHeight: 300)
            } else {
                ContentView()
                    .onAppear {
                        logger.info("GrafApp: ContentView appeared")
                    }
                    .frame(minWidth: 800, minHeight: 600)
                    .onDisappear {
                        logger.info("GrafApp: ContentView disappeared")
                    }
            }
        }
    }
}
