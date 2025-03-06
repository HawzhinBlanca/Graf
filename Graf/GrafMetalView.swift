import AppKit
import MetalKit
import SwiftUI

// MARK: - GrafMetalView
struct GrafMetalView: NSViewRepresentable {
    var renderer: GrafRenderer
    var isDebugMode: Bool

    // MARK: - NSViewRepresentable

    public func makeNSView(context: Context) -> MTKView {
        // Create the Metal view
        let view = MTKView()

        // Check for Metal support
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("ERROR: Metal is not supported on this device")
            return createErrorView(message: "Metal is not supported on this device")
        }

        // Configure the view
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.05, alpha: 1.0)

        // Configure the renderer
        renderer.setupMetal(device: device, view: view)

        // Set the renderer as the view's delegate
        view.delegate = renderer

        // Set up gesture recognizers
        let panGesture = NSPanGestureRecognizer(
            target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        view.addGestureRecognizer(panGesture)

        let magnificationGesture = NSMagnificationGestureRecognizer(
            target: context.coordinator, action: #selector(Coordinator.handleMagnification(_:)))
        view.addGestureRecognizer(magnificationGesture)

        let clickGesture = NSClickGestureRecognizer(
            target: context.coordinator, action: #selector(Coordinator.handleClick(_:)))
        view.addGestureRecognizer(clickGesture)

        return view
    }

    public func updateNSView(_ nsView: MTKView, context: Context) {
        // Update view if needed
        context.coordinator.isDebugMode = isDebugMode
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator(renderer: renderer, isDebugMode: isDebugMode)
    }

    // MARK: - Helper Methods

    private func createErrorView(message: String) -> MTKView {
        // Create a basic MTKView (it won't render anything)
        let view = MTKView(frame: .zero)

        // Create a text label for the error message
        let label = NSTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 100))
        label.stringValue = "Error: \(message)"
        label.textColor = NSColor.white
        label.backgroundColor = NSColor.red
        label.isBezeled = false
        label.drawsBackground = true
        label.isEditable = false
        label.isSelectable = false
        label.alignment = .center

        // Add the label to the view
        view.addSubview(label)

        return view
    }

    // MARK: - Coordinator

    public class Coordinator: NSObject {
        var renderer: GrafRenderer
        var isDebugMode: Bool

        init(renderer: GrafRenderer, isDebugMode: Bool) {
            self.renderer = renderer
            self.isDebugMode = isDebugMode
            super.init()
        }

        @objc func handlePan(_ gesture: NSPanGestureRecognizer) {
            let translation = gesture.translation(in: gesture.view)

            // Forward to renderer
            renderer.handleMouseDrag(dx: Float(translation.x), dy: Float(translation.y))

            // Reset translation for continuous updates
            gesture.setTranslation(.zero, in: gesture.view)
        }

        @objc func handleMagnification(_ gesture: NSMagnificationGestureRecognizer) {
            // Forward to renderer for zoom
            renderer.zoom(factor: Float(1.0 + gesture.magnification))

            // Reset for continuous updates
            gesture.magnification = 0
        }

        @objc func handleClick(_ gesture: NSClickGestureRecognizer) {
            if let view = gesture.view as? MTKView {
                let location = gesture.location(in: view)
                renderer.handleMouseClick(at: location, size: view.bounds.size, view: view)
            }
        }
    }
}
