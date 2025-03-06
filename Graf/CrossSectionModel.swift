import Foundation
import MetalKit
import SwiftUI
import simd


/// Model for the cross-section view
class CrossSectionModel: ObservableObject {
    // Metal renderer
    var renderer: CrossSectionRenderer!

    // Cross-section system
    private var crossSection: GA4DCrossSection

    // 4D object
    private var vertices4D: [SIMD4<Float>]
    private var edges: [(Int, Int)]
    private var faces: [[Int]]

    // Animation
    private var animationTimer: Timer?
    private var animationStartTime: TimeInterval = 0
    private var animationStartValue: Float = 0
    private var animationEndValue: Float = 0
    private var animationDuration: TimeInterval = 0
    private var animationLoop: Bool = false

    // Display settings
    @Published var showEdges: Bool = true
    @Published var showFaces: Bool = true

    // Add a property to store the current plane
    private var currentPlane: GA4DCrossSection.HyperplaneType = .wConstant(0)

    // Set the cross-section plane
    func setCrossSectionPlane(_ plane: GA4DCrossSection.HyperplaneType) {
        currentPlane = plane  // Update our locally stored currentPlane
        crossSection.setHyperplane(plane)
        updateCrossSection()
    }

    // Get the current cross-section plane
    func getCurrentPlane() -> GA4DCrossSection.HyperplaneType {
        return currentPlane
    }

    // Initializer
    init(vertices: [SIMD4<Float>], edges: [(Int, Int)], faces: [[Int]] = []) {
        self.vertices4D = vertices
        self.edges = edges
        self.faces = faces

        // Initialize cross-section system
        self.crossSection = GA4DCrossSection(vertices: vertices, edges: edges)
    }

    // Setup Metal renderer
    func setupRenderer(device: MTLDevice, mtkView: MTKView) {
        self.renderer = CrossSectionRenderer(
            device: device,
            metalView: mtkView,
            vertices: vertices4D,
            edges: edges,
            faces: faces
        )

        // Generate initial cross-section
        updateCrossSection()
    }

    // Update the 4D vertices (e.g., after rotation)
    func updateVertices(_ vertices: [SIMD4<Float>]) {
        self.vertices4D = vertices
        crossSection.updateVertices(vertices)
        updateCrossSection()
    }

    // Update cross-section
    private func updateCrossSection() {
        let section = crossSection.generateCrossSection()
        renderer.updateCrossSection(section, showEdges: showEdges, showFaces: showFaces)
    }

    // Start animation of cross-section
    func startAnimation(
        startValue: Float, endValue: Float, duration: TimeInterval, loop: Bool = false
    ) {
        // Stop any existing animation
        stopAnimation()

        // Set animation parameters
        animationStartValue = startValue
        animationEndValue = endValue
        animationDuration = duration
        animationLoop = loop
        animationStartTime = Date.timeIntervalSinceReferenceDate

        // Start animation timer
        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) {
            [weak self] _ in
            self?.updateAnimation()
        }
    }

    // Stop animation
    func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
    }

    // Update animation frame
    private func updateAnimation() {
        let currentTime = Date.timeIntervalSinceReferenceDate
        var elapsed = currentTime - animationStartTime

        // Handle looping
        if animationLoop {
            elapsed = elapsed.truncatingRemainder(dividingBy: animationDuration)
        } else if elapsed > animationDuration {
            // Animation complete
            stopAnimation()
            return
        }

        // Calculate current value
        let progress = Float(elapsed / animationDuration)
        let value = animationStartValue + progress * (animationEndValue - animationStartValue)

        // Update hyperplane
        switch currentPlane {
        case .xConstant:
            setCrossSectionPlane(.xConstant(value))
        case .yConstant:
            setCrossSectionPlane(.yConstant(value))
        case .zConstant:
            setCrossSectionPlane(.zConstant(value))
        case .wConstant:
            setCrossSectionPlane(.wConstant(value))
        case .custom(let normal, _):
            setCrossSectionPlane(.custom(normal, value))
        }
    }

    // Camera controls
    func rotateCamera(dx: Float, dy: Float) {
        renderer.rotateCamera(dx: dx, dy: dy)
    }

    func zoomCamera(factor: Float) {
        renderer.zoomCamera(factor: factor)
    }

    func resetCamera() {
        renderer.resetCamera()
    }

    // Set the color method
    func setColorMethod(_ method: GA4DVisualizer.ColorMethod) {
        renderer.setColorMethod(method)
        updateCrossSection()
    }
}
