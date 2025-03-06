import AppKit
import MetalKit
import SwiftUI

/// View that allows interactive exploration of 4D cross-sections
struct CrossSectionView: View {
    // The 4D object to cross-section
    @ObservedObject var model: CrossSectionModel

    // Controls
    @State private var hyperplaneValue: Float = 0.0
    @State private var hyperplaneAxis: GA4DCrossSection.HyperplaneType = .wConstant(0)
    @State private var isAnimating: Bool = false
    @State private var animationSpeed: Float = 1.0
    @State private var showEdges: Bool = true
    @State private var showFaces: Bool = true
    @State private var colorMode: GA4DVisualizer.ColorMethod = .wCoordinate

    var body: some View {
        VStack {
            Spacer()

            // 3D cross-section view
            MetalCrossSectionView(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipped()
                .overlay(
                    AxisIndicatorView(hyperplaneAxis: $hyperplaneAxis)
                        .frame(width: 100, height: 100)
                        .padding(),
                    alignment: .topTrailing
                )
                .overlay(
                    Text("Value: \(hyperplaneValue, specifier: "%.2f")")
                        .padding(4)
                        .background(Color.black.opacity(0.5))
                        .foregroundColor(.white)
                        .cornerRadius(4)
                        .padding(),
                    alignment: .topLeading
                )

            // Debug status text
            Text("Cross-section of 4D Object")
                .font(.caption)
                .padding(.horizontal)

            // Control panel
            VStack(spacing: 12) {
                // Hyperplane selection
                HStack {
                    Text("Hyperplane:")

                    Picker("Hyperplane", selection: $hyperplaneAxis) {
                        Text("X").tag(GA4DCrossSection.HyperplaneType.xConstant(hyperplaneValue))
                        Text("Y").tag(GA4DCrossSection.HyperplaneType.yConstant(hyperplaneValue))
                        Text("Z").tag(GA4DCrossSection.HyperplaneType.zConstant(hyperplaneValue))
                        Text("W").tag(GA4DCrossSection.HyperplaneType.wConstant(hyperplaneValue))
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .onChange(of: hyperplaneAxis) { oldValue, newValue in
                        updateHyperplane()
                    }

                    Spacer()

                    Button("Reset View") {
                        model.resetCamera()
                    }
                }

                // Hyperplane value slider
                HStack {
                    Text("Value:")
                    Slider(value: $hyperplaneValue, in: -1...1, step: 0.01)
                        .onChange(of: hyperplaneValue) { oldValue, newValue in
                            updateHyperplane()
                        }
                    Text("\(hyperplaneValue, specifier: "%.2f")")
                        .frame(width: 50, alignment: .trailing)
                        .monospacedDigit()
                }

                // Animation controls
                HStack {
                    Toggle("Animate", isOn: $isAnimating)
                        .onChange(of: isAnimating) { oldValue, newValue in
                            if newValue {
                                startAnimation()
                            } else {
                                stopAnimation()
                            }
                        }

                    if isAnimating {
                        Text("Speed:")
                        Slider(value: $animationSpeed, in: 0.1...3.0, step: 0.1)
                            .frame(width: 100)
                    }

                    Spacer()
                }

                // Visualization options
                HStack {
                    Toggle("Show Edges", isOn: $showEdges)
                        .onChange(of: showEdges) { oldValue, newValue in
                            model.showEdges = showEdges
                        }

                    Toggle("Show Faces", isOn: $showFaces)
                        .onChange(of: showFaces) { oldValue, newValue in
                            model.showFaces = showFaces
                        }

                    Picker("Color", selection: $colorMode) {
                        Text("W-Coordinate").tag(GA4DVisualizer.ColorMethod.wCoordinate)
                        Text("Distance").tag(GA4DVisualizer.ColorMethod.distance)
                        Text("Normal").tag(GA4DVisualizer.ColorMethod.normal)
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: colorMode) { oldValue, newValue in
                        model.setColorMethod(colorMode)
                    }
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
        }
        .onAppear {
            // Initialize with current values
            updateHyperplane()
            model.showEdges = showEdges
            model.showFaces = showFaces
            model.setColorMethod(colorMode)
        }
    }

    // Update the hyperplane based on selected axis and value
    private func updateHyperplane() {
        switch hyperplaneAxis {
        case .xConstant:
            model.setCrossSectionPlane(.xConstant(hyperplaneValue))
        case .yConstant:
            model.setCrossSectionPlane(.yConstant(hyperplaneValue))
        case .zConstant:
            model.setCrossSectionPlane(.zConstant(hyperplaneValue))
        case .wConstant:
            model.setCrossSectionPlane(.wConstant(hyperplaneValue))
        case .custom:
            // For custom planes, keep the existing normal and just update the distance
            if case let .custom(normal, _) = model.getCurrentPlane() {
                model.setCrossSectionPlane(.custom(normal, hyperplaneValue))
            }
        }
    }

    // Start cross-section animation
    private func startAnimation() {
        // Create animation parameters
        let startValue: Float = -1.0
        let endValue: Float = 1.0
        let duration: Double = 5.0 / Double(animationSpeed)

        // Start the animation
        model.startAnimation(
            startValue: startValue,
            endValue: endValue,
            duration: duration,
            loop: true
        )
    }

    // Stop cross-section animation
    private func stopAnimation() {
        model.stopAnimation()
    }
}

/// Metal view for rendering the cross-section
struct MetalCrossSectionView: NSViewRepresentable {
    @ObservedObject var model: CrossSectionModel

    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()

        // Set up Metal
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        mtkView.device = device
        mtkView.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .depth32Float

        // Set up renderer
        model.setupRenderer(device: device, mtkView: mtkView)
        mtkView.delegate = model.renderer

        // Set up gesture recognizers
        let panGesture = NSPanGestureRecognizer(
            target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        mtkView.addGestureRecognizer(panGesture)

        let magnificationGesture = NSMagnificationGestureRecognizer(
            target: context.coordinator, action: #selector(Coordinator.handleMagnification(_:)))
        mtkView.addGestureRecognizer(magnificationGesture)

        return mtkView
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        // Update view if needed
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(model: model)
    }

    class Coordinator: NSObject {
        var model: CrossSectionModel

        init(model: CrossSectionModel) {
            self.model = model
        }

        @objc func handlePan(_ gesture: NSPanGestureRecognizer) {
            let translation = gesture.translation(in: gesture.view)

            // Rotate camera
            model.rotateCamera(dx: Float(translation.x), dy: Float(translation.y))

            // Reset translation for continuous updates
            gesture.setTranslation(.zero, in: gesture.view)
        }

        @objc func handleMagnification(_ gesture: NSMagnificationGestureRecognizer) {
            // Zoom camera
            model.zoomCamera(factor: Float(1.0 + gesture.magnification))

            // Reset magnification for continuous updates
            gesture.magnification = 0
        }
    }
}

/// Small view showing the current cutting axis
struct AxisIndicatorView: View {
    @Binding var hyperplaneAxis: GA4DCrossSection.HyperplaneType

    var body: some View {
        ZStack {
            // Background
            Circle()
                .fill(Color.black.opacity(0.3))

            // X-axis
            Line(from: CGPoint(x: 25, y: 50), to: CGPoint(x: 75, y: 50))
                .stroke(Color.red, lineWidth: isXAxis ? 3 : 1)
            Text("X")
                .foregroundColor(.red)
                .position(x: 85, y: 50)

            // Y-axis
            Line(from: CGPoint(x: 50, y: 25), to: CGPoint(x: 50, y: 75))
                .stroke(Color.green, lineWidth: isYAxis ? 3 : 1)
            Text("Y")
                .foregroundColor(.green)
                .position(x: 50, y: 15)

            // Z-axis
            Line(from: CGPoint(x: 35, y: 35), to: CGPoint(x: 65, y: 65))
                .stroke(Color.blue, lineWidth: isZAxis ? 3 : 1)
            Text("Z")
                .foregroundColor(.blue)
                .position(x: 72, y: 72)

            // W-axis (shown as a circle)
            Circle()
                .stroke(Color.purple, lineWidth: isWAxis ? 3 : 1)
                .frame(width: 30, height: 30)
            Text("W")
                .foregroundColor(.purple)
                .position(x: 30, y: 30)
        }
    }

    private var isXAxis: Bool {
        if case .xConstant = hyperplaneAxis { return true }
        return false
    }

    private var isYAxis: Bool {
        if case .yConstant = hyperplaneAxis { return true }
        return false
    }

    private var isZAxis: Bool {
        if case .zConstant = hyperplaneAxis { return true }
        return false
    }

    private var isWAxis: Bool {
        if case .wConstant = hyperplaneAxis { return true }
        return false
    }
}

/// Line shape for drawing
struct Line: Shape {
    var from: CGPoint
    var to: CGPoint

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: from)
        path.addLine(to: to)
        return path
    }
}
