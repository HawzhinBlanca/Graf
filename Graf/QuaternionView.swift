//
//  QuaternionView.swift
//  Graf
//
//  Created by HAWZHIN on 04/03/2024.
//

import MetalKit
import SwiftUI
import simd

/// A view that provides interactive quaternion visualization in the style of 3Blue1Brown
struct QuaternionView: View {
    // State for quaternion components
    @State private var quaternionW: Float = 1.0
    @State private var quaternionX: Float = 0.0
    @State private var quaternionY: Float = 0.0
    @State private var quaternionZ: Float = 0.0

    // Visualization options
    @State private var showAxisLabels: Bool = true
    @State private var showUnitSphere: Bool = true
    @State private var showRotationArcs: Bool = true
    @State private var animateRotation: Bool = false
    @State private var rotationSpeed: Float = 0.5

    // Interaction state
    @State private var showInputSphere: Bool = true
    @State private var showOutputSphere: Bool = true
    @State private var viewMode: QuaternionViewMode = .dualSphere

    // Pre-defined quaternion examples
    @State private var selectedExample: Int = 0

    // Renderer reference
    private let renderer = QuaternionRenderer()

    // Example quaternions
    private let examples = [
        ("Identity", SIMD4<Float>(1, 0, 0, 0)),
        ("X-Axis 90°", SIMD4<Float>(0.7071, 0.7071, 0, 0)),
        ("Y-Axis 90°", SIMD4<Float>(0.7071, 0, 0.7071, 0)),
        ("Z-Axis 90°", SIMD4<Float>(0.7071, 0, 0, 0.7071)),
        ("X-Y Diagonal", SIMD4<Float>(0.7071, 0.5, 0.5, 0)),
        ("Complex Rotation", SIMD4<Float>(0.5, 0.5, 0.5, 0.5)),
    ]

    var body: some View {
        VStack {
            Text("3Blue1Brown-Style Quaternion Visualization")
                .font(.headline)
                .padding(.top)
            // Visualization view
            QuaternionMetalView(renderer: renderer)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black.opacity(0.1))
                .border(Color.gray, width: 1)
                .padding(.horizontal)
                .onAppear {
                    updateQuaternion()
                }
            // Control panel
            TabView {
                // Components tab
                VStack(spacing: 12) {
                    Text("Quaternion Components")
                        .font(.headline)
                    HStack {
                        Text("w: \(quaternionW, specifier: "%.2f")")
                            .frame(width: 80, alignment: .leading)
                        Slider(value: $quaternionW, in: -1...1, step: 0.01)
                            .onChange(of: quaternionW) { _, _ in
                                normalizeQuaternion()
                                updateQuaternion()
                            }
                    }
                    HStack {
                        Text("x: \(quaternionX, specifier: "%.2f")")
                            .frame(width: 80, alignment: .leading)
                        Slider(value: $quaternionX, in: -1...1, step: 0.01)
                            .onChange(of: quaternionX) { _, _ in
                                normalizeQuaternion()
                                updateQuaternion()
                            }
                    }
                    HStack {
                        Text("y: \(quaternionY, specifier: "%.2f")")
                            .frame(width: 80, alignment: .leading)
                        Slider(value: $quaternionY, in: -1...1, step: 0.01)
                            .onChange(of: quaternionY) { _, _ in
                                normalizeQuaternion()
                                updateQuaternion()
                            }
                    }
                    HStack {
                        Text("z: \(quaternionZ, specifier: "%.2f")")
                            .frame(width: 80, alignment: .leading)
                        Slider(value: $quaternionZ, in: -1...1, step: 0.01)
                            .onChange(of: quaternionZ) { _, _ in
                                normalizeQuaternion()
                                updateQuaternion()
                            }
                    }

                    Divider()

                    Text("Rotation Angle: \(getRotationAngle(), specifier: "%.0f")°")
                    Text(
                        "Rotation Axis: [\(getRotationAxis().x, specifier: "%.2f"), \(getRotationAxis().y, specifier: "%.2f"), \(getRotationAxis().z, specifier: "%.2f")]"
                    )

                    Divider()

                    HStack {
                        Text("Example:")
                        Picker("Example", selection: $selectedExample) {
                            ForEach(0..<examples.count, id: \.self) { i in
                                Text(examples[i].0).tag(i)
                            }
                        }
                        .onChange(of: selectedExample) { _, newValue in
                            let example = examples[newValue]
                            quaternionW = example.1.x
                            quaternionX = example.1.y
                            quaternionY = example.1.z
                            quaternionZ = example.1.w
                            updateQuaternion()
                        }
                    }
                    Button("Reset to Identity") {
                        quaternionW = 1.0
                        quaternionX = 0.0
                        quaternionY = 0.0
                        quaternionZ = 0.0
                        updateQuaternion()
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                .tabItem {
                    Label("Components", systemImage: "slider.horizontal.3")
                }
                // Visualization Options tab
                VStack(spacing: 12) {
                    Text("Visualization Options")
                        .font(.headline)
                    Picker("View Mode", selection: $viewMode) {
                        Text("Dual Sphere").tag(QuaternionViewMode.dualSphere)
                        Text("Single Sphere").tag(QuaternionViewMode.singleSphere)
                        Text("Rotation Map").tag(QuaternionViewMode.rotationMap)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .onChange(of: viewMode) { _, newValue in
                        renderer.setViewMode(viewMode: newValue)
                    }
                    Toggle("Show Axis Labels", isOn: $showAxisLabels)
                        .onChange(of: showAxisLabels) { _, newValue in
                            renderer.setShowAxisLabels(show: newValue)
                        }
                    Toggle("Show Unit Sphere", isOn: $showUnitSphere)
                        .onChange(of: showUnitSphere) { _, newValue in
                            renderer.setShowUnitSphere(show: newValue)
                        }
                    Toggle("Show Rotation Arcs", isOn: $showRotationArcs)
                        .onChange(of: showRotationArcs) { _, newValue in
                            renderer.setShowRotationArcs(show: newValue)
                        }

                    Divider()

                    Toggle("Animate Rotation", isOn: $animateRotation)
                        .onChange(of: animateRotation) { _, newValue in
                            renderer.setAnimateRotation(animate: newValue)
                        }

                    if animateRotation {
                        HStack {
                            Text("Speed: \(rotationSpeed, specifier: "%.1f")")
                                .frame(width: 100, alignment: .leading)
                            Slider(value: $rotationSpeed, in: 0.1...2.0, step: 0.1)
                                .onChange(of: rotationSpeed) { _, newValue in
                                    renderer.setRotationSpeed(speed: newValue)
                                }
                        }
                    }

                    Divider()

                    Toggle("Show Input Sphere", isOn: $showInputSphere)
                        .onChange(of: showInputSphere) { _, newValue in
                            renderer.setShowInputSphere(show: newValue)
                        }

                    Toggle("Show Output Sphere", isOn: $showOutputSphere)
                        .onChange(of: showOutputSphere) { _, newValue in
                            renderer.setShowOutputSphere(show: newValue)
                        }
                }
                .padding()
                .tabItem {
                    Label("Options", systemImage: "gear")
                }
                // Educational tab
                VStack(spacing: 15) {
                    Text("Understanding Quaternions")
                        .font(.headline)
                    Text("Quaternions represent rotations in 3D space using four components:")
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Text("q = w + xi + yj + zk")
                        .font(.system(.body, design: .monospaced))
                        .padding(.vertical, 5)

                    Text("where:")
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Text("• w is the scalar (real) part")
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text("• x, y, z form the vector (imaginary) part")
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Divider()

                    Text("For unit quaternions (|q| = 1):")
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Text("• Rotation angle = 2 × acos(w)")
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text("• Rotation axis = normalize(x, y, z)")
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Spacer()

                    Link(
                        "Learn more from 3Blue1Brown",
                        destination: URL(string: "https://youtu.be/zjMuIxRvygQ")!)
                }
                .padding()
                .tabItem {
                    Label("Learn", systemImage: "book")
                }
            }
            .frame(height: 250)
            .padding(.horizontal)
        }
    }

    /// Normalize the quaternion to ensure it's a unit quaternion
    private func normalizeQuaternion() {
        let length = sqrt(
            quaternionW * quaternionW + quaternionX * quaternionX + quaternionY * quaternionY
                + quaternionZ * quaternionZ)

        if length > 0.0001 {
            quaternionW /= length
            quaternionX /= length
            quaternionY /= length
            quaternionZ /= length
        } else {
            // Default to identity quaternion if length is too small
            quaternionW = 1.0
            quaternionX = 0.0
            quaternionY = 0.0
            quaternionZ = 0.0
        }
    }

    /// Update the renderer with the current quaternion
    private func updateQuaternion() {
        let quaternion = SIMD4<Float>(quaternionW, quaternionX, quaternionY, quaternionZ)
        renderer.setQuaternion(quaternion: quaternion)
    }

    /// Get the rotation angle in degrees
    private func getRotationAngle() -> Float {
        return 2.0 * acos(max(min(quaternionW, 1.0), -1.0)) * 180.0 / Float.pi
    }

    /// Get the normalized rotation axis
    private func getRotationAxis() -> SIMD3<Float> {
        let axis = SIMD3<Float>(quaternionX, quaternionY, quaternionZ)
        let len = length(axis)

        if len > 0.0001 {
            return axis / len
        } else {
            // Default to x-axis if vector part is too small
            return SIMD3<Float>(1, 0, 0)
        }
    }
}

/// The Metal view for quaternion visualization
struct QuaternionMetalView: NSViewRepresentable {
    var renderer: QuaternionRenderer

    func makeNSView(context: Context) -> MTKView {
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
        view.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.05, alpha: 1.0)

        // Set up the renderer with Metal
        renderer.setupMetal(device: device, view: view)

        // Set the renderer as the view's delegate
        view.delegate = renderer

        // Add gesture recognizers for interaction
        let panGesture = NSPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePan(_:)))
        view.addGestureRecognizer(panGesture)

        let magnificationGesture = NSMagnificationGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleMagnification(_:)))
        view.addGestureRecognizer(magnificationGesture)

        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        // No updates needed
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(renderer: renderer)
    }

    private func createErrorView(message: String) -> MTKView {
        let view = MTKView(frame: .zero)

        let label = NSTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 100))
        label.stringValue = "Error: \(message)"
        label.textColor = NSColor.white
        label.backgroundColor = NSColor.red
        label.isBezeled = false
        label.drawsBackground = true
        label.isEditable = false
        label.isSelectable = false
        label.alignment = .center

        view.addSubview(label)

        return view
    }

    class Coordinator: NSObject {
        var renderer: QuaternionRenderer

        init(renderer: QuaternionRenderer) {
            self.renderer = renderer
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
    }
}

/// View mode for quaternion visualization
enum QuaternionViewMode {
    case dualSphere  // Show both input and output spheres
    case singleSphere  // Show single sphere with rotation
    case rotationMap  // Show the quaternion as a map from S³ to SO(3)
}

/// Renders quaternion visualizations in Metal
class QuaternionRenderer: NSObject, MTKViewDelegate {
    // MARK: - Metal Resources
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var pipelineState: MTLRenderPipelineState?
    private var depthState: MTLDepthStencilState?

    // MARK: - Rendering Resources
    private var sphereVertexBuffer: MTLBuffer?
    private var sphereIndexBuffer: MTLBuffer?
    private var axisVertexBuffer: MTLBuffer?
    private var axisIndexBuffer: MTLBuffer?
    private var arcVertexBuffer: MTLBuffer?
    private var sphereIndexCount: Int = 0
    private var axisIndexCount: Int = 0
    private var arcVertexCount: Int = 0

    // MARK: - Camera
    private var viewMatrix = matrix_identity_float4x4
    private var projectionMatrix = matrix_identity_float4x4
    private var camera = Camera()

    // MARK: - Quaternion State
    private var quaternion = SIMD4<Float>(1, 0, 0, 0)  // Identity quaternion (w,x,y,z)
    private var rotationMatrix = matrix_identity_float4x4

    // MARK: - Visualization Options
    private var viewMode: QuaternionViewMode = .dualSphere
    private var showAxisLabels: Bool = true
    private var showUnitSphere: Bool = true
    private var showRotationArcs: Bool = true
    private var animateRotation: Bool = false
    private var rotationSpeed: Float = 0.5
    private var showInputSphere: Bool = true
    private var showOutputSphere: Bool = true

    // MARK: - Animation State
    private var animationAngle: Float = 0.0
    private var startTime: TimeInterval = Date().timeIntervalSinceReferenceDate

    // MARK: - Uniforms Structure
    struct Uniforms {
        var modelMatrix: matrix_float4x4
        var viewMatrix: matrix_float4x4
        var projectionMatrix: matrix_float4x4
        var normalMatrix: matrix_float3x3
        var color: SIMD4<Float>
        var options: SIMD4<UInt32>  // Packed options
    }

    // MARK: - Initialization
    override init() {
        super.init()

        // Initialize with default settings
        startTime = Date().timeIntervalSinceReferenceDate
        camera.position = SIMD3<Float>(0, 0, -3)

        // Initialize rotation matrix
        updateRotationMatrix()
    }

    // MARK: - Metal Setup
    func setupMetal(device: MTLDevice, view: MTKView) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()

        // Set up depth stencil state
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        self.depthState = device.makeDepthStencilState(descriptor: depthDescriptor)

        // Set up the rendering pipeline
        setupRenderPipeline(device: device, view: view)

        // Create geometry for rendering
        createSphereGeometry()
        createAxisGeometry()
        createArcGeometry()
    }

    private func setupRenderPipeline(device: MTLDevice, view: MTKView) {
        do {
            // Get default library containing Metal shaders
            guard let library = device.makeDefaultLibrary() else {
                print("ERROR: Failed to load default Metal library")
                return
            }

            // Get vertex and fragment shader functions
            guard let vertexFunction = library.makeFunction(name: "vertexShader"),
                let fragmentFunction = library.makeFunction(name: "fragmentShader")
            else {
                print("ERROR: Failed to find shader functions")
                return
            }

            // Create render pipeline descriptor
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.label = "Quaternion Pipeline"
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat

            // Configure blending for transparency
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor =
                .oneMinusSourceAlpha

            // Set depth format
            pipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat

            // Configure vertex descriptor
            let vertexDescriptor = MTLVertexDescriptor()

            // Position attribute (float3)
            vertexDescriptor.attributes[0].format = .float3
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0

            // Normal attribute (float3)
            vertexDescriptor.attributes[1].format = .float3
            vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
            vertexDescriptor.attributes[1].bufferIndex = 0

            // Color attribute (float4)
            vertexDescriptor.attributes[2].format = .float4
            vertexDescriptor.attributes[2].offset = MemoryLayout<SIMD3<Float>>.stride * 2
            vertexDescriptor.attributes[2].bufferIndex = 0

            // Layout configuration
            vertexDescriptor.layouts[0].stride =
                MemoryLayout<SIMD3<Float>>.stride * 2 + MemoryLayout<SIMD4<Float>>.stride
            vertexDescriptor.layouts[0].stepRate = 1
            vertexDescriptor.layouts[0].stepFunction = .perVertex

            // Assign the vertex descriptor to the pipeline
            pipelineDescriptor.vertexDescriptor = vertexDescriptor

            // Create pipeline state
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("ERROR: Failed to create pipeline state: \(error)")
        }
    }

    // MARK: - Geometry Creation

    /// Create the sphere geometry
    private func createSphereGeometry() {
        guard let device = device else { return }

        let resolution = 32
        let radius: Float = 1.0

        var vertices: [Vertex] = []
        var indices: [UInt32] = []

        // Generate sphere vertices
        for i in 0...resolution {
            let phi = Float.pi * Float(i) / Float(resolution)
            let sinPhi = sin(phi)
            let cosPhi = cos(phi)

            for j in 0...resolution {
                let theta = 2.0 * Float.pi * Float(j) / Float(resolution)
                let sinTheta = sin(theta)
                let cosTheta = cos(theta)

                // Calculate position
                let x = radius * sinPhi * cosTheta
                let y = radius * sinPhi * sinTheta
                let z = radius * cosPhi

                // Calculate normal (same as position for unit sphere)
                let normal = normalize(SIMD3<Float>(x, y, z))

                // Color based on position
                let color = SIMD4<Float>(
                    (normal.x + 1.0) * 0.5,
                    (normal.y + 1.0) * 0.5,
                    (normal.z + 1.0) * 0.5,
                    1.0
                )

                // Add vertex
                vertices.append(
                    Vertex(
                        position: SIMD3<Float>(x, y, z),
                        normal: normal,
                        color: color
                    ))
            }
        }

        // Generate indices for triangles
        for i in 0..<resolution {
            for j in 0..<resolution {
                let first = i * (resolution + 1) + j
                let second = first + resolution + 1

                // First triangle
                indices.append(UInt32(first))
                indices.append(UInt32(first + 1))
                indices.append(UInt32(second))

                // Second triangle
                indices.append(UInt32(second))
                indices.append(UInt32(first + 1))
                indices.append(UInt32(second + 1))
            }
        }

        // Create buffers
        sphereVertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<Vertex>.stride,
            options: .storageModeShared
        )

        sphereIndexBuffer = device.makeBuffer(
            bytes: indices,
            length: indices.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        sphereIndexCount = indices.count
    }

    /// Create coordinate axes geometry
    private func createAxisGeometry() {
        guard let device = device else { return }

        let axisLength: Float = 1.5

        let vertices: [Vertex] = [
            // X-axis (red)
            Vertex(
                position: SIMD3<Float>(0, 0, 0),
                normal: SIMD3<Float>(1, 0, 0),
                color: SIMD4<Float>(1, 0, 0, 1)
            ),
            Vertex(
                position: SIMD3<Float>(axisLength, 0, 0),
                normal: SIMD3<Float>(1, 0, 0),
                color: SIMD4<Float>(1, 0, 0, 1)
            ),
            // Y-axis (green)
            Vertex(
                position: SIMD3<Float>(0, 0, 0),
                normal: SIMD3<Float>(0, 1, 0),
                color: SIMD4<Float>(0, 1, 0, 1)
            ),
            Vertex(
                position: SIMD3<Float>(0, axisLength, 0),
                normal: SIMD3<Float>(0, 1, 0),
                color: SIMD4<Float>(0, 1, 0, 1)
            ),
            // Z-axis (blue)
            Vertex(
                position: SIMD3<Float>(0, 0, 0),
                normal: SIMD3<Float>(0, 0, 1),
                color: SIMD4<Float>(0, 0, 1, 1)
            ),
            Vertex(
                position: SIMD3<Float>(0, 0, axisLength),
                normal: SIMD3<Float>(0, 0, 1),
                color: SIMD4<Float>(0, 0, 1, 1)
            ),
        ]

        let indices: [UInt32] = [
            0, 1,  // X-axis
            2, 3,  // Y-axis
            4, 5,  // Z-axis
        ]

        // Create buffers
        axisVertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<Vertex>.stride,
            options: .storageModeShared
        )

        axisIndexBuffer = device.makeBuffer(
            bytes: indices,
            length: indices.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        axisIndexCount = indices.count
    }

    /// Create arc geometry for rotation visualization
    private func createArcGeometry() {
        guard let device = device else { return }

        let segments = 64
        let radius: Float = 1.0

        var vertices: [Vertex] = []

        // Create an arc representing a rotation (unit circle in XY plane)
        for i in 0...segments {
            let angle = Float.pi * 2.0 * Float(i) / Float(segments)
            let x = radius * cos(angle)
            let y = radius * sin(angle)

            let position = SIMD3<Float>(x, y, 0)
            let normal = SIMD3<Float>(0, 0, 1)
            let color = SIMD4<Float>(1, 0.5, 0, 1)  // Orange arc

            vertices.append(
                Vertex(
                    position: position,
                    normal: normal,
                    color: color
                ))
        }

        // Create buffer
        arcVertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<Vertex>.stride,
            options: .storageModeShared
        )

        arcVertexCount = vertices.count
    }

    // MARK: - Quaternion Operations

    /// Set the current quaternion
    func setQuaternion(quaternion: SIMD4<Float>) {
        self.quaternion = quaternion
        updateRotationMatrix()
    }

    /// Convert quaternion to rotation matrix
    private func updateRotationMatrix() {
        // Normalize the quaternion to ensure it's a unit quaternion
        let q = normalize(quaternion)

        // Extract quaternion components
        let w = q.x
        let x = q.y
        let y = q.z
        let z = q.w

        // Calculate rotation matrix from quaternion
        rotationMatrix = matrix_float4x4(
            SIMD4<Float>(
                1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0),
            SIMD4<Float>(
                2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x, 0),
            SIMD4<Float>(
                2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }

    // MARK: - Visualization Options

    /// Set the view mode
    func setViewMode(viewMode: QuaternionViewMode) {
        self.viewMode = viewMode
    }

    /// Show or hide axis labels
    func setShowAxisLabels(show: Bool) {
        self.showAxisLabels = show
    }

    /// Show or hide the unit sphere
    func setShowUnitSphere(show: Bool) {
        self.showUnitSphere = show
    }

    /// Show or hide rotation arcs
    func setShowRotationArcs(show: Bool) {
        self.showRotationArcs = show
    }

    /// Enable or disable rotation animation
    func setAnimateRotation(animate: Bool) {
        self.animateRotation = animate
    }

    /// Set the rotation animation speed
    func setRotationSpeed(speed: Float) {
        self.rotationSpeed = speed
    }

    /// Show or hide the input sphere
    func setShowInputSphere(show: Bool) {
        self.showInputSphere = show
    }

    /// Show or hide the output sphere
    func setShowOutputSphere(show: Bool) {
        self.showOutputSphere = show
    }

    // MARK: - Interaction Handlers

    /// Handle mouse drag for camera rotation
    func handleMouseDrag(dx: Float, dy: Float) {
        camera.rotateY += dx * 0.01
        camera.rotateX += dy * 0.01
    }

    /// Handle zoom
    func zoom(factor: Float) {
        camera.zoom *= factor
        camera.zoom = max(0.1, min(camera.zoom, 10.0))  // Clamp zoom
    }

    // MARK: - MTKViewDelegate Methods

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Update camera aspect ratio
        let aspect = Float(size.width / size.height)
        camera.aspectRatio = aspect

        // Update projection matrix
        projectionMatrix = camera.projectionMatrix(type: .perspective)
    }

    func draw(in view: MTKView) {
        guard device != nil,
            let commandQueue = commandQueue,
            let pipelineState = pipelineState,
            let renderPassDescriptor = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable
        else {
            return
        }

        // Update animation
        let currentTime = Float(Date().timeIntervalSinceReferenceDate - startTime)
        if animateRotation {
            animationAngle = currentTime * rotationSpeed
            // Create an animated quaternion
            let angle = animationAngle
            let axis = SIMD3<Float>(sin(angle * 0.7), cos(angle * 0.5), sin(angle * 0.3))
            let normalizedAxis = normalize(axis)

            // Convert to quaternion (w, x, y, z)
            let halfAngle = angle * 0.5
            let sinHalfAngle = sin(halfAngle)
            quaternion = SIMD4<Float>(
                cos(halfAngle),
                normalizedAxis.x * sinHalfAngle,
                normalizedAxis.y * sinHalfAngle,
                normalizedAxis.z * sinHalfAngle
            )

            updateRotationMatrix()
        }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        // Create render command encoder
        guard
            let renderEncoder = commandBuffer.makeRenderCommandEncoder(
                descriptor: renderPassDescriptor)
        else {
            return
        }

        // Set depth state
        if let depthState = depthState {
            renderEncoder.setDepthStencilState(depthState)
        }

        // Set render pipeline state
        renderEncoder.setRenderPipelineState(pipelineState)

        // Update view matrix
        viewMatrix = camera.viewMatrix()

        // Draw based on view mode
        switch viewMode {
        case .dualSphere:
            drawDualSphereView(renderEncoder: renderEncoder)
        case .singleSphere:
            drawSingleSphereView(renderEncoder: renderEncoder)
        case .rotationMap:
            drawRotationMapView(renderEncoder: renderEncoder)
        }

        // End encoding and submit
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: - Rendering Methods

    /// Draw the dual sphere visualization (input and output spheres)
    private func drawDualSphereView(renderEncoder: MTLRenderCommandEncoder) {
        // Draw input sphere on the left
        if showInputSphere {
            let inputTransform = matrix_multiply(
                matrix_float4x4(translation: SIMD3<Float>(-1.5, 0, 0)),
                matrix_float4x4(scaling: SIMD3<Float>(0.8, 0.8, 0.8))
            )
            // Draw sphere
            if showUnitSphere {
                drawSphere(renderEncoder: renderEncoder, transform: inputTransform)
            }
            // Draw axes
            if showAxisLabels {
                drawAxes(renderEncoder: renderEncoder, transform: inputTransform)
            }
        }

        // Draw output sphere on the right
        if showOutputSphere {
            let outputTransform = matrix_multiply(
                matrix_multiply(
                    matrix_float4x4(translation: SIMD3<Float>(1.5, 0, 0)),
                    rotationMatrix
                ),
                matrix_float4x4(scaling: SIMD3<Float>(0.8, 0.8, 0.8))
            )
            // Draw sphere
            if showUnitSphere {
                drawSphere(renderEncoder: renderEncoder, transform: outputTransform)
            }
            // Draw axes
            if showAxisLabels {
                drawAxes(renderEncoder: renderEncoder, transform: outputTransform)
            }
        }

        // Draw rotation arc
        if showRotationArcs {
            // Get quaternion angle and axis
            let angle = 2.0 * acos(max(min(quaternion.x, 1.0), -1.0))
            var axis = SIMD3<Float>(quaternion.y, quaternion.z, quaternion.w)

            if length(axis) < 0.001 {
                axis = SIMD3<Float>(1, 0, 0)  // Default to x-axis
            } else {
                axis = normalize(axis)
            }

            // Create transformation to align arc with rotation axis
            let alignmentTransform = createRotationAlignmentMatrix(fromZ: axis)

            // Scale arc based on rotation angle
            let arcTransform = matrix_multiply(
                matrix_multiply(
                    matrix_float4x4(translation: SIMD3<Float>(0, 0, 0)),
                    alignmentTransform
                ),
                matrix_float4x4(scaling: SIMD3<Float>(0.8, 0.8, 0.8))
            )

            drawArc(renderEncoder: renderEncoder, transform: arcTransform, angle: angle)
        }
    }

    /// Draw the single sphere visualization
    private func drawSingleSphereView(renderEncoder: MTLRenderCommandEncoder) {
        // Draw the sphere
        if showUnitSphere {
            drawSphere(renderEncoder: renderEncoder, transform: matrix_identity_float4x4)
        }

        // Draw the axes
        if showAxisLabels {
            // Draw the original axes
            drawAxes(
                renderEncoder: renderEncoder, transform: matrix_identity_float4x4,
                color: SIMD4<Float>(0.5, 0.5, 0.5, 0.5))

            // Draw the rotated axes
            drawAxes(renderEncoder: renderEncoder, transform: rotationMatrix)
        }

        // Draw rotation arc
        if showRotationArcs {
            // Get quaternion angle and axis
            let angle = 2.0 * acos(max(min(quaternion.x, 1.0), -1.0))
            var axis = SIMD3<Float>(quaternion.y, quaternion.z, quaternion.w)

            if length(axis) < 0.001 {
                axis = SIMD3<Float>(1, 0, 0)  // Default to x-axis
            } else {
                axis = normalize(axis)
            }

            // Create transformation to align arc with rotation axis
            let alignmentTransform = createRotationAlignmentMatrix(fromZ: axis)

            drawArc(renderEncoder: renderEncoder, transform: alignmentTransform, angle: angle)
        }
    }

    /// Draw the rotation map visualization
    private func drawRotationMapView(renderEncoder: MTLRenderCommandEncoder) {
        // Draw the sphere
        if showUnitSphere {
            drawSphere(renderEncoder: renderEncoder, transform: matrix_identity_float4x4)
        }

        // Draw the axes
        if showAxisLabels {
            drawAxes(renderEncoder: renderEncoder, transform: matrix_identity_float4x4)
        }

        // Draw a point representing the current quaternion
        let w = quaternion.x
        let x = quaternion.y
        let y = quaternion.z
        let z = quaternion.w

        // Scale the point based on w (4D to 3D)
        // Here we use a simple stereographic projection
        let scale: Float = 2.0
        let projection = scale / (1.0 + w)
        let pointPosition = SIMD3<Float>(
            x * projection,
            y * projection,
            z * projection
        )

        // Draw a small sphere at the quaternion position
        let pointTransform = matrix_multiply(
            matrix_float4x4(translation: pointPosition),
            matrix_float4x4(scaling: SIMD3<Float>(0.1, 0.1, 0.1))
        )

        // Use a specific color for the point
        let pointColor = SIMD4<Float>(1.0, 0.3, 0.3, 1.0)
        drawSphere(renderEncoder: renderEncoder, transform: pointTransform, color: pointColor)

        // Draw rotation arc if needed
        if showRotationArcs {
            let angle = 2.0 * acos(max(min(quaternion.x, 1.0), -1.0))
            var axis = SIMD3<Float>(quaternion.y, quaternion.z, quaternion.w)

            if length(axis) < 0.001 {
                axis = SIMD3<Float>(1, 0, 0)
            } else {
                axis = normalize(axis)
            }

            let alignmentTransform = createRotationAlignmentMatrix(fromZ: axis)
            drawArc(renderEncoder: renderEncoder, transform: alignmentTransform, angle: angle)
        }
    }

    // MARK: - Drawing Primitives

    /// Draw a sphere
    private func drawSphere(
        renderEncoder: MTLRenderCommandEncoder, transform: matrix_float4x4,
        color: SIMD4<Float>? = nil
    ) {
        guard let sphereVertexBuffer = sphereVertexBuffer,
            let sphereIndexBuffer = sphereIndexBuffer
        else {
            return
        }

        // Set vertex buffer
        renderEncoder.setVertexBuffer(sphereVertexBuffer, offset: 0, index: 0)

        // Create normal matrix
        let normalMatrix = simd_float3x3(
            SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )

        // Create uniforms
        var uniforms = Uniforms(
            modelMatrix: transform,
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            normalMatrix: normalMatrix,
            color: color ?? SIMD4<Float>(1, 1, 1, 1),
            options: SIMD4<UInt32>(0, 0, 0, 0)
        )

        // Set uniforms
        renderEncoder.setVertexBytes(
            &uniforms,
            length: MemoryLayout<Uniforms>.stride,
            index: 1
        )

        // Draw indexed triangles
        renderEncoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: sphereIndexCount,
            indexType: .uint32,
            indexBuffer: sphereIndexBuffer,
            indexBufferOffset: 0
        )
    }

    /// Draw coordinate axes
    private func drawAxes(
        renderEncoder: MTLRenderCommandEncoder, transform: matrix_float4x4,
        color: SIMD4<Float>? = nil
    ) {
        guard let axisVertexBuffer = axisVertexBuffer,
            let axisIndexBuffer = axisIndexBuffer
        else {
            return
        }

        // Set vertex buffer
        renderEncoder.setVertexBuffer(axisVertexBuffer, offset: 0, index: 0)

        // Create normal matrix
        let normalMatrix = simd_float3x3(
            SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )

        // Create uniforms
        var uniforms = Uniforms(
            modelMatrix: transform,
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            normalMatrix: normalMatrix,
            color: color ?? SIMD4<Float>(1, 1, 1, 1),
            options: SIMD4<UInt32>(1, 0, 0, 0)  // Option 0 set to 1 to indicate axes
        )

        // Set uniforms
        renderEncoder.setVertexBytes(
            &uniforms,
            length: MemoryLayout<Uniforms>.stride,
            index: 1
        )

        // Draw indexed lines
        renderEncoder.drawIndexedPrimitives(
            type: .line,
            indexCount: axisIndexCount,
            indexType: .uint32,
            indexBuffer: axisIndexBuffer,
            indexBufferOffset: 0
        )
    }

    /// Draw a rotation arc
    private func drawArc(
        renderEncoder: MTLRenderCommandEncoder, transform: matrix_float4x4, angle: Float
    ) {
        guard let arcVertexBuffer = arcVertexBuffer else {
            return
        }

        // Set vertex buffer
        renderEncoder.setVertexBuffer(arcVertexBuffer, offset: 0, index: 0)

        // Create normal matrix
        let normalMatrix = simd_float3x3(
            SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )

        // Create uniforms
        var uniforms = Uniforms(
            modelMatrix: transform,
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            normalMatrix: normalMatrix,
            color: SIMD4<Float>(1, 0.5, 0, 1),  // Orange arc
            options: SIMD4<UInt32>(2, 0, 0, 0)  // Option 0 set to 2 to indicate arc
        )

        // Set uniforms
        renderEncoder.setVertexBytes(
            &uniforms,
            length: MemoryLayout<Uniforms>.stride,
            index: 1
        )

        // Calculate the number of vertices to draw based on angle
        let segmentCount = Int(Float(arcVertexCount) * angle / (Float.pi * 2))
        let vertexCount = min(max(2, segmentCount), arcVertexCount)

        // Draw line strip
        renderEncoder.drawPrimitives(
            type: .lineStrip,
            vertexStart: 0,
            vertexCount: vertexCount
        )
    }

    // MARK: - Helper Methods

    /// Create a rotation matrix that aligns the z-axis with the given vector
    private func createRotationAlignmentMatrix(fromZ targetVector: SIMD3<Float>) -> matrix_float4x4
    {
        let zAxis = SIMD3<Float>(0, 0, 1)
        let normalizedTarget = normalize(targetVector)

        // Special case: if target is already aligned with Z
        if abs(dot(normalizedTarget, zAxis) - 1) < 0.0001 {
            return matrix_identity_float4x4
        }

        // Special case: if target is opposite to Z
        if abs(dot(normalizedTarget, zAxis) + 1) < 0.0001 {
            return matrix_float4x4(
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(0, 0, -1, 0),
                SIMD4<Float>(0, 0, 0, 1)
            )
        }

        // General case: calculate rotation axis and angle
        let rotationAxis = normalize(cross(zAxis, normalizedTarget))
        let rotationAngle = acos(dot(zAxis, normalizedTarget))

        // Convert to quaternion
        let halfAngle = rotationAngle * 0.5
        let sinHalfAngle = sin(halfAngle)

        let q = SIMD4<Float>(
            cos(halfAngle),
            rotationAxis.x * sinHalfAngle,
            rotationAxis.y * sinHalfAngle,
            rotationAxis.z * sinHalfAngle
        )

        // Convert quaternion to matrix
        let x = q.y
        let y = q.z
        let z = q.w
        let w = q.x

        return matrix_float4x4(
            SIMD4<Float>(
                1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0),
            SIMD4<Float>(
                2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x, 0),
            SIMD4<Float>(
                2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }
}

// MARK: - Vertex Structure
struct Vertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var color: SIMD4<Float>
}
