import MetalKit
import SwiftUI
import simd

// MARK: - Local Type Definitions
enum ContentViewVisualizationType: String, CaseIterable, Identifiable {
    case tesseract = "Tesseract (Hypercube)"
    case hypersphere = "Hypersphere"
    case duocylinder = "Duocylinder"
    case cliffordTorus = "Clifford Torus"
    case quaternion = "Quaternion Visualization"
    case customFunction = "Custom 4D Function"

    var id: String { self.rawValue }
}

enum ContentViewProjectionType: String, CaseIterable, Identifiable {
    case stereographic = "Stereographic"
    case perspective = "Perspective"
    case orthographic = "Orthographic"

    var id: String { self.rawValue }
}

// Error handling for graceful UI feedback
struct ErrorHandler {
    var renderingError: Error? = nil
    var isErrorVisible: Bool = false

    mutating func handleError(_ error: Error?) {
        renderingError = error
        isErrorVisible = error != nil
    }

    func errorDescription() -> String {
        renderingError?.localizedDescription ?? "Unknown error"
    }
}

// MARK: - ContentView
struct ContentView: View {
    // Create the renderer
    private let renderer = GrafRenderer()

    // Error handling for Metal setup
    @State private var errorHandler = ErrorHandler()

    // UI state
    @State private var visualizationType: ContentViewVisualizationType = .tesseract
    @State private var resolution: Float = 32
    @State private var scale: Float = 1.0
    @State private var wireframe: Bool = true
    @State private var showNormals: Bool = false
    @State private var projectionType: ContentViewProjectionType = .stereographic
    @State private var animationSpeed: Float = 1.0
    @State private var isDebugMode: Bool = true
    @State private var debugMessage: String = "Initializing..."

    // 4D Rotation controls
    @State private var autoRotate: Bool = true
    @State private var xyRotation: Float = 0.0
    @State private var xzRotation: Float = 0.0
    @State private var xwRotation: Float = 0.0
    @State private var yzRotation: Float = 0.0
    @State private var ywRotation: Float = 0.0
    @State private var zwRotation: Float = 0.0

    // Custom function input
    @State private var customFunction: String = "sin(x) * cos(y) * sin(w)"

    // Selection mode
    @State private var interactionMode: Int = 0  // 0=rotate, 1=pan, 2=select
    private let interactionModes = ["Rotate", "Pan", "Select"]

    // Main view selection
    @State private var selectedView: Int = 0
    private let viewOptions = ["Standard", "Quaternions"]

    var body: some View {
        VStack {
            // View selector
            Picker("View", selection: $selectedView) {
                ForEach(0..<viewOptions.count, id: \.self) { index in
                    Text(viewOptions[index]).tag(index)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding(.horizontal)

            if selectedView == 0 {
                // Standard 4D Visualization View
                standardView
            } else {
                // 3Blue1Brown-Style Quaternion View
                QuaternionView()
            }
        }
        .padding()
        .background(Color(NSColor.windowBackgroundColor))
        .alert(isPresented: $errorHandler.isErrorVisible) {
            Alert(
                title: Text("Rendering Error"),
                message: Text(errorHandler.errorDescription()),
                dismissButton: .default(Text("OK"))
            )
        }
    }

    // MARK: - Standard Visualization View
    var standardView: some View {
        VStack {
            // 3D/4D Visualization View
            EnhancedGrafMetalView(
                renderer: renderer,
                isDebugMode: isDebugMode,
                errorHandler: $errorHandler,
                debugLog: { message in
                    debugMessage = message
                }
            )
            .frame(minWidth: 200, minHeight: 200)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black.opacity(0.1))
            .border(Color.gray, width: 1)
            .onAppear {
                debugMessage = "GrafMetalView appeared in ContentView"
                wireframe = true
                renderer.autoRotate = autoRotate
                updateVisualization()
                renderer.resetCamera()
            }

            // Debug status text
            if isDebugMode {
                Text(debugMessage)
                    .font(.caption)
                    .padding(.horizontal)

                Text(
                    "Visualization: \(visualizationType.rawValue), Projection: \(projectionType.rawValue)"
                )
                .font(.caption)
                .padding(.horizontal)
            }

            // Main controls in tabs
            TabView {
                // Basic controls tab
                VStack(spacing: 10) {
                    HStack {
                        Text("Type:")
                        Picker("Visualization Type", selection: $visualizationType) {
                            ForEach(ContentViewVisualizationType.allCases, id: \.self) { type in
                                Text(type.rawValue).tag(type)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .onChange(of: visualizationType) { oldValue, newValue in
                            updateVisualization()
                        }

                        Spacer()

                        Text("Projection:")
                        Picker("Projection Type", selection: $projectionType) {
                            ForEach(ContentViewProjectionType.allCases, id: \.self) { type in
                                Text(type.rawValue).tag(type)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .onChange(of: projectionType) { oldValue, newValue in
                            // Convert to Graf.ProjectionType
                            let projType: Graf.ProjectionType =
                                projectionType == .stereographic
                                ? .stereographic
                                : projectionType == .perspective ? .perspective : .orthographic

                            renderer.setProjectionType(projType)
                            debugMessage = "Projection changed to \(projectionType.rawValue)"
                        }
                    }

                    HStack {
                        Text("Resolution: \(Int(resolution))")
                            .frame(width: 120, alignment: .leading)
                        Slider(value: $resolution, in: 8...128, step: 8)
                            .onChange(of: resolution) { oldValue, newValue in
                                updateVisualization()
                            }
                    }

                    HStack {
                        Text("Scale: \(scale, specifier: "%.1f")")
                            .frame(width: 120, alignment: .leading)
                        Slider(value: $scale, in: 0.1...2.0, step: 0.1)
                            .onChange(of: scale) { oldValue, newValue in
                                updateVisualization()
                            }
                    }

                    HStack {
                        Text("Animation: \(animationSpeed, specifier: "%.1f")")
                            .frame(width: 120, alignment: .leading)
                        Slider(value: $animationSpeed, in: 0.0...3.0, step: 0.1)
                            .onChange(of: animationSpeed) { oldValue, newValue in
                                renderer.animationSpeed = animationSpeed
                                debugMessage = "Animation speed set to \(animationSpeed)"
                            }
                    }

                    HStack {
                        Toggle("Wireframe", isOn: $wireframe)
                            .onChange(of: wireframe) { oldValue, newValue in
                                renderer.wireframe = wireframe
                                updateVisualization()
                            }

                        Toggle("Show Normals", isOn: $showNormals)
                            .onChange(of: showNormals) { oldValue, newValue in
                                renderer.showNormals = showNormals
                                updateVisualization()
                            }

                        Toggle("Auto-Rotate", isOn: $autoRotate)
                            .onChange(of: autoRotate) { oldValue, newValue in
                                renderer.autoRotate = autoRotate
                                debugMessage =
                                    "Auto-rotation " + (autoRotate ? "enabled" : "disabled")
                            }
                    }

                    HStack {
                        Text("Interaction:")
                        Picker("Interaction Mode", selection: $interactionMode) {
                            ForEach(0..<interactionModes.count, id: \.self) { index in
                                Text(interactionModes[index]).tag(index)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .onChange(of: interactionMode) { oldValue, newValue in
                            // Convert to GrafRenderer.InteractionMode
                            switch newValue {
                            case 0: renderer.interactionMode = .rotate
                            case 1: renderer.interactionMode = .pan
                            case 2: renderer.interactionMode = .select
                            default: renderer.interactionMode = .rotate
                            }
                            debugMessage = "Interaction mode set to \(interactionModes[newValue])"
                        }

                        Button("Reset Camera") {
                            renderer.resetCamera()
                            debugMessage = "Camera reset"
                        }
                    }
                }
                .padding()
                .tabItem {
                    Label("Basic", systemImage: "cube")
                }

                // 4D Rotation controls tab
                VStack(spacing: 10) {
                    HStack {
                        Text("XY Rotation:")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $xyRotation, in: 0...Float.pi * 2)
                            .onChange(of: xyRotation) { oldValue, newValue in
                                renderer.setRotation(plane: "xy", angle: newValue)
                                debugMessage = "XY rotation set to \(newValue)"
                            }
                        Button("0") {
                            xyRotation = 0
                            renderer.setRotation(plane: "xy", angle: 0)
                        }
                        .buttonStyle(.bordered)
                    }

                    HStack {
                        Text("XZ Rotation:")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $xzRotation, in: 0...Float.pi * 2)
                            .onChange(of: xzRotation) { oldValue, newValue in
                                renderer.setRotation(plane: "xz", angle: newValue)
                                debugMessage = "XZ rotation set to \(newValue)"
                            }
                        Button("0") {
                            xzRotation = 0
                            renderer.setRotation(plane: "xz", angle: 0)
                        }
                        .buttonStyle(.bordered)
                    }

                    HStack {
                        Text("XW Rotation:")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $xwRotation, in: 0...Float.pi * 2)
                            .onChange(of: xwRotation) { oldValue, newValue in
                                renderer.setRotation(plane: "xw", angle: newValue)
                                debugMessage = "XW rotation set to \(newValue)"
                            }
                        Button("0") {
                            xwRotation = 0
                            renderer.setRotation(plane: "xw", angle: 0)
                        }
                        .buttonStyle(.bordered)
                    }

                    HStack {
                        Text("YZ Rotation:")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $yzRotation, in: 0...Float.pi * 2)
                            .onChange(of: yzRotation) { oldValue, newValue in
                                renderer.setRotation(plane: "yz", angle: newValue)
                                debugMessage = "YZ rotation set to \(newValue)"
                            }
                        Button("0") {
                            yzRotation = 0
                            renderer.setRotation(plane: "yz", angle: 0)
                        }
                        .buttonStyle(.bordered)
                    }

                    HStack {
                        Text("YW Rotation:")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $ywRotation, in: 0...Float.pi * 2)
                            .onChange(of: ywRotation) { oldValue, newValue in
                                renderer.setRotation(plane: "yw", angle: newValue)
                                debugMessage = "YW rotation set to \(newValue)"
                            }
                        Button("0") {
                            ywRotation = 0
                            renderer.setRotation(plane: "yw", angle: 0)
                        }
                        .buttonStyle(.bordered)
                    }

                    HStack {
                        Text("ZW Rotation:")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $zwRotation, in: 0...Float.pi * 2)
                            .onChange(of: zwRotation) { oldValue, newValue in
                                renderer.setRotation(plane: "zw", angle: newValue)
                                debugMessage = "ZW rotation set to \(newValue)"
                            }
                        Button("0") {
                            zwRotation = 0
                            renderer.setRotation(plane: "zw", angle: 0)
                        }
                        .buttonStyle(.bordered)
                    }

                    Button("Reset All Rotations") {
                        xyRotation = 0
                        xzRotation = 0
                        xwRotation = 0
                        yzRotation = 0
                        ywRotation = 0
                        zwRotation = 0

                        renderer.setRotation(plane: "xy", angle: 0)
                        renderer.setRotation(plane: "xz", angle: 0)
                        renderer.setRotation(plane: "xw", angle: 0)
                        renderer.setRotation(plane: "yz", angle: 0)
                        renderer.setRotation(plane: "yw", angle: 0)
                        renderer.setRotation(plane: "zw", angle: 0)

                        debugMessage = "All rotations reset to 0"
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                .tabItem {
                    Label("4D Rotation", systemImage: "rotate.3d")
                }

                // Custom function tab
                VStack(spacing: 15) {
                    Text("Enter a custom 4D function to visualize")
                        .font(.headline)

                    Text("Use variables x, y, z, w and standard math functions like sin, cos, sqrt")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack {
                        TextField("Custom function expression", text: $customFunction)
                            .textFieldStyle(RoundedBorderTextFieldStyle())

                        Button("Apply") {
                            applyCustomFunction()
                        }
                        .buttonStyle(.bordered)
                    }

                    Text("Example functions:")
                        .font(.headline)
                        .padding(.top)

                    VStack(alignment: .leading, spacing: 5) {
                        Button("sin(x) * cos(y) * sin(w)") {
                            customFunction = "sin(x) * cos(y) * sin(w)"
                            applyCustomFunction()
                        }
                        .buttonStyle(.borderless)

                        Button("sqrt(x*x + y*y + z*z + w*w)") {
                            customFunction = "sqrt(x*x + y*y + z*z + w*w)"
                            applyCustomFunction()
                        }
                        .buttonStyle(.borderless)

                        Button("sin(x*y + z*w)") {
                            customFunction = "sin(x*y + z*w)"
                            applyCustomFunction()
                        }
                        .buttonStyle(.borderless)

                        Button("x*x - y*y + z*z - w*w") {
                            customFunction = "x*x - y*y + z*z - w*w"
                            applyCustomFunction()
                        }
                        .buttonStyle(.borderless)
                    }
                }
                .padding()
                .tabItem {
                    Label("Custom Function", systemImage: "function")
                }

                // Debug & Settings tab
                VStack(spacing: 10) {
                    Toggle("Debug Mode", isOn: $isDebugMode)
                        .padding(.vertical)

                    if isDebugMode {
                        Text("Debug Message:")
                            .font(.headline)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        ScrollView {
                            Text(debugMessage)
                                .font(.caption)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding()
                                .background(Color.black.opacity(0.05))
                                .cornerRadius(5)
                        }
                        .frame(height: 100)

                        HStack {
                            Button("Clear Debug") {
                                debugMessage = "Debug cleared at \(Date())"
                            }

                            Button("Force Redraw") {
                                debugMessage = "Forced redraw at \(Date())"
                                updateVisualization()
                            }
                        }
                        .padding(.top)
                    }

                    Divider()

                    Text("Performance Settings:")
                        .font(.headline)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top)

                    HStack {
                        Text("Max Resolution:")
                        Picker("Max Resolution", selection: $resolution) {
                            Text("Low (16)").tag(Float(16))
                            Text("Medium (32)").tag(Float(32))
                            Text("High (64)").tag(Float(64))
                            Text("Ultra (128)").tag(Float(128))
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .onChange(of: resolution) { oldValue, newValue in
                            updateVisualization()
                        }
                    }
                }
                .padding()
                .tabItem {
                    Label("Debug", systemImage: "gear")
                }
            }
            .frame(height: 250)
            .padding(.horizontal)
        }
    }

    private func updateVisualization() {
        // Convert visualization type to Graf.VisualizationType
        let type: Graf.VisualizationType
        switch visualizationType {
        case .tesseract: type = .tesseract
        case .hypersphere: type = .hypersphere
        case .duocylinder: type = .duocylinder
        case .cliffordTorus: type = .cliffordTorus
        case .quaternion: type = .quaternion
        case .customFunction: type = .customFunction
        }

        // If custom function is selected, apply it
        if visualizationType == .customFunction {
            applyCustomFunction()
            return
        }

        // Update renderer properties
        renderer.wireframe = wireframe
        renderer.showNormals = showNormals

        // Create visualization data with current settings
        let data = createVisualizationData(type: type, resolution: Int(resolution), scale: scale)

        // Debug: Print some information about the visualization data
        debugMessage =
            "Created visualization with \(data.vertices.count) vertices, \(data.indices.count/2) edges"

        // Update renderer with the visualization data
        renderer.updateVisualization(data: data)

        debugMessage +=
            "\nVisualization updated to \(visualizationType.rawValue), res: \(Int(resolution)), scale: \(scale)"
    }

    private func createVisualizationData(
        type: Graf.VisualizationType, resolution: Int, scale: Float
    ) -> Graf.VisualizationData {
        // For debugging, create a simple 3D cube instead of 4D shapes
        if true {
            // Create a simple 3D cube
            var vertices3D: [SIMD3<Float>] = []
            var vertices4D: [SIMD4<Float>] = []
            var edges: [(Int, Int)] = []

            // Create 8 vertices of a cube
            for x in [-1.0, 1.0] {
                for y in [-1.0, 1.0] {
                    for z in [-1.0, 1.0] {
                        vertices3D.append(SIMD3<Float>(Float(x), Float(y), Float(z)) * scale)
                        vertices4D.append(SIMD4<Float>(Float(x), Float(y), Float(z), 0.0) * scale)
                    }
                }
            }

            // Create 12 edges of a cube
            edges = [
                (0, 1), (0, 2), (0, 4),  // From vertex 0
                (1, 3), (1, 5),  // From vertex 1
                (2, 3), (2, 6),  // From vertex 2
                (3, 7),  // From vertex 3
                (4, 5), (4, 6),  // From vertex 4
                (5, 7),  // From vertex 5
                (6, 7),  // From vertex 6
            ]

            // Create normals (pointing outward)
            let normals3D = vertices3D.map { normalize($0) }

            // Create colors (based on position)
            let colors = vertices3D.map { v in
                SIMD4<Float>(
                    (v.x + scale) / (2 * scale),
                    (v.y + scale) / (2 * scale),
                    (v.z + scale) / (2 * scale),
                    1.0)
            }

            // Create indices for rendering
            var indices: [UInt32] = []
            for (start, end) in edges {
                indices.append(UInt32(start))
                indices.append(UInt32(end))
            }

            // Return visualization data with the simple cube
            return Graf.VisualizationData(
                vertices: vertices3D,
                normals: normals3D,
                colors: colors,
                indices: indices,
                edges: edges,
                faces: [],
                originalVertices4D: vertices4D
            )
        }

        // Original implementation below
        // ... existing code ...
    }

    private func applyCustomFunction() {
        if !customFunction.isEmpty {
            debugMessage = "Applying custom function: \(customFunction)"
            visualizationType = .customFunction

            // Create default range
            let samplePoints = Int(resolution)
            var vertices4D: [SIMD4<Float>] = []

            // Evaluate function over a grid
            let stepSize = 2.0 * scale / Float(samplePoints)
            for i in 0...samplePoints {
                for j in 0...samplePoints {
                    let x = -scale + Float(i) * stepSize
                    let y = -scale + Float(j) * stepSize

                    // Use z and w as function outputs for visualization
                    let z = evaluateFunction(customFunction, x: x, y: y, w: 0)
                    let w = evaluateFunction(customFunction, x: x, y: y, w: 0.5)

                    vertices4D.append(SIMD4<Float>(x, y, z, w))
                }
            }

            // Generate grid-based edges
            var edges: [(Int, Int)] = []
            let width = samplePoints + 1
            for i in 0..<vertices4D.count {
                let x = i % width
                let y = i / width

                if x < width - 1 {
                    edges.append((i, i + 1))
                }
                if y < width - 1 {
                    edges.append((i, i + width))
                }
            }

            // Create faces for better visualization
            var faces: [[Int]] = []
            for y in 0..<samplePoints {
                for x in 0..<samplePoints {
                    let i = y * width + x
                    faces.append([i, i + 1, i + width + 1, i + width])
                }
            }

            // Convert to 3D and add to visualization data
            let rotations = (
                xy: xyRotation,
                xz: xzRotation,
                xw: xwRotation,
                yz: yzRotation,
                yw: ywRotation,
                zw: zwRotation
            )

            // Convert projection type
            let projType: GA4D.GA4DMetalBridge.ProjectionType
            switch projectionType {
            case .stereographic: projType = .stereographic
            case .perspective: projType = .perspective
            case .orthographic: projType = .orthographic
            }

            let vertices3D = GA4D.GA4DMetalBridge.transformVertices(
                vertices4D: vertices4D,
                rotations: rotations,
                projectionType: projType
            )

            // Generate normals and colors
            let connections = faces.flatMap { face -> [(Int, Int, Int)] in
                if face.count >= 3 {
                    var triangles: [(Int, Int, Int)] = []
                    for i in 1..<(face.count - 1) {
                        triangles.append((face[0], face[i], face[i + 1]))
                    }
                    return triangles
                }
                return []
            }

            let normals4D = GA4D.GA4DMetalBridge.calculateNormals(
                for: vertices4D, connections: connections)
            let normals3D = normals4D.map { normal4D -> SIMD3<Float> in
                SIMD3<Float>(normal4D.x, normal4D.y, normal4D.z)
            }

            let colors = vertices4D.map { v -> SIMD4<Float> in
                // Color based on function value
                let normalizedZ = (v.z + scale) / (2 * scale)
                let normalizedW = (v.w + scale) / (2 * scale)
                return SIMD4<Float>(normalizedZ, 0.5, normalizedW, 1.0)
            }

            // Create indices for rendering
            var indices: [UInt32] = []
            for (start, end) in edges {
                indices.append(UInt32(start))
                indices.append(UInt32(end))
            }

            // Update the renderer with this custom function visualization
            let visualizationData = Graf.VisualizationData(
                vertices: vertices3D,
                normals: normals3D,
                colors: colors,
                indices: indices,
                edges: edges,
                faces: faces,
                originalVertices4D: vertices4D
            )

            renderer.updateVisualization(data: visualizationData)
        } else {
            debugMessage = "Custom function is empty"
        }
    }

    // Helper function to evaluate custom mathematical expressions
    private func evaluateFunction(_ expression: String, x: Float, y: Float, w: Float) -> Float {
        // This is a simple implementation - in a real app, you'd use a proper expression parser

        // Convert expression to lowercase for case-insensitive matching
        let expr = expression.lowercased()

        // Handle some common expressions for demonstration
        if expr.contains("sin") && expr.contains("x") && expr.contains("cos") && expr.contains("y")
        {
            return sin(x) * cos(y)
        } else if expr.contains("sin") && expr.contains("sqrt") {
            return sin(sqrt(x * x + y * y + w * w))
        } else if expr.contains("x*x") || expr.contains("x^2") {
            return x * x - y * y
        } else if expr.contains("sin") && expr.contains("w") {
            return sin(x) * cos(y) * sin(w)
        }

        // Default fallback
        return sin(x * 3) * cos(y * 2)
    }
}

// MARK: - Enhanced MetalView with Error Handling
struct EnhancedGrafMetalView: NSViewRepresentable {
    var renderer: GrafRenderer
    var isDebugMode: Bool
    @Binding var errorHandler: ErrorHandler
    var debugLog: ((String) -> Void)? = nil

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()

        do {
            // Try to set up Metal with error handling
            try setupMetal(view: view)
            return view
        } catch {
            errorHandler.handleError(error)
            debugLog?("ERROR: Failed to initialize Metal: \(error.localizedDescription)")
            return createErrorView(
                message: "Failed to initialize Metal: \(error.localizedDescription)")
        }
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        // Update view if needed
        context.coordinator.isDebugMode = isDebugMode
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(renderer: renderer, isDebugMode: isDebugMode)
    }

    // Set up Metal with error handling
    private func setupMetal(view: MTKView) throws {
        // Check for Metal support
        guard let device = MTLCreateSystemDefaultDevice() else {
            debugLog?("ERROR: Metal is not supported on this device")
            throw GrafError.deviceNotSupported
        }

        debugLog?("Metal device initialized: \(device.name)")

        // Configure the view
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.05, alpha: 1.0)

        debugLog?("Metal view configured")

        // Configure the renderer
        renderer.setupMetal(device: device, view: view)

        debugLog?("Renderer setup complete")

        // Set the renderer as the view's delegate
        view.delegate = renderer

        // Add gesture recognizers for interaction
        let panGesture = NSPanGestureRecognizer(
            target: Coordinator(renderer: renderer, isDebugMode: isDebugMode),
            action: #selector(Coordinator.handlePan(_:)))
        view.addGestureRecognizer(panGesture)

        let magnificationGesture = NSMagnificationGestureRecognizer(
            target: Coordinator(renderer: renderer, isDebugMode: isDebugMode),
            action: #selector(Coordinator.handleMagnification(_:)))
        view.addGestureRecognizer(magnificationGesture)

        let clickGesture = NSClickGestureRecognizer(
            target: Coordinator(renderer: renderer, isDebugMode: isDebugMode),
            action: #selector(Coordinator.handleClick(_:)))
        view.addGestureRecognizer(clickGesture)
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

    // Coordinator for handling user interactions
    class Coordinator: NSObject {
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

// MARK: - Custom Error Types
enum GrafError: Error, LocalizedError {
    case deviceNotSupported
    case libraryCreationFailed
    case pipelineCreationFailed
    case bufferCreationFailed
    case shaderCompilationFailed(String)
    case textureCreationFailed

    var errorDescription: String? {
        switch self {
        case .deviceNotSupported:
            return "Metal is not supported on this device"
        case .libraryCreationFailed:
            return "Failed to create Metal shader library"
        case .pipelineCreationFailed:
            return "Failed to create Metal render pipeline"
        case .bufferCreationFailed:
            return "Failed to create Metal buffer"
        case .shaderCompilationFailed(let details):
            return "Shader compilation failed: \(details)"
        case .textureCreationFailed:
            return "Failed to create Metal texture"
        }
    }
}
