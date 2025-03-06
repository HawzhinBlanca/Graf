import Foundation
import Metal
import MetalKit
import simd

/// The main renderer for Graf application, handling Metal setup, rendering, and 4D geometry
public class GrafRenderer: NSObject, MTKViewDelegate {
    // MARK: - Metal Resources

    /// The Metal device for GPU operations
    public private(set) var device: MTLDevice?

    /// Command queue for submitting work to the GPU
    public private(set) var commandQueue: MTLCommandQueue?

    /// The render pipeline state for standard rendering
    public private(set) var pipelineState: MTLRenderPipelineState?

    /// The render pipeline state for wireframe rendering
    private var wireframePipelineState: MTLRenderPipelineState?

    /// The render pipeline state for points rendering
    private var pointsPipelineState: MTLRenderPipelineState?

    /// The depth stencil state for proper depth testing
    private var depthState: MTLDepthStencilState?

    /// Sampler state for texture operations
    private var samplerState: MTLSamplerState?

    /// Property for backward compatibility
    public var pipeline: MTLRenderPipelineState? { return pipelineState }

    /// Debug message for status reporting
    private var debugMessage: String = ""

    // MARK: - Render Settings

    /// Camera for controlling view and projection
    public var camera = Camera()

    /// Timestamp for tracking when the last update occurred
    private var lastUpdateTime: TimeInterval = Date().timeIntervalSinceReferenceDate

    /// Buffer for uniform data used in shaders
    private var uniformBuffer: MTLBuffer?

    /// Flag for wireframe rendering mode
    public var wireframe: Bool = true {
        didSet {
            if oldValue != wireframe {
                // Regenerate indices if wireframe mode changes
                updateBuffers()
            }
        }
    }

    /// Flag for showing vertex normals
    public var showNormals: Bool = false {
        didSet {
            if oldValue != showNormals {
                // Update visualization if normal display changes
                updateCurrentVisualization()
            }
        }
    }

    /// Animation speed multiplier
    public var animationSpeed: Float = 1.0

    /// Projection type (perspective, orthographic, stereographic)
    public var projectionType: ProjectionType = .stereographic {
        didSet {
            if oldValue != projectionType {
                // Update projection matrix when type changes
                updateProjectionMatrix()
            }
        }
    }

    /// Current visualization type
    private var currentVisType: VisualizationType = .tesseract

    /// Current scale factor for visualization
    private var currentScale: Float = 1.0

    /// Current resolution for visualization
    private var currentResolution: Int = 32

    /// Whether to auto-rotate the 4D object
    public var autoRotate: Bool = true

    /// Whether to use smooth shading
    public var smoothShading: Bool = true

    /// Current custom function expression
    private var customFunctionExpression: String = "sin(x) * cos(y) * sin(w)"

    /// Parser for mathematical expressions
    private var expressionParser = ExpressionParser()

    // MARK: - Interactivity State

    /// Interaction mode for the controller
    public enum InteractionMode {
        case rotate
        case pan
        case select
        case sketch
    }

    /// Current interaction mode
    public var interactionMode: InteractionMode = .rotate

    /// Whether interaction is currently happening
    private var isInteracting: Bool = false

    /// Last mouse position for interaction
    private var lastMousePosition: CGPoint = .zero

    /// Selected vertex index for editing
    private var selectedVertexIndex: Int = -1

    // MARK: - Rotation state for 4D
    private var rotationAngles: (xy: Float, xz: Float, xw: Float, yz: Float, yw: Float, zw: Float) =
        (0, 0, 0, 0, 0, 0)

    /// Manual rotation overrides
    private var manualRotation: (xy: Float, xz: Float, xw: Float, yz: Float, yw: Float, zw: Float) =
        (0, 0, 0, 0, 0, 0)
    // MARK: - Buffers and Geometry

    /// Metal vertex buffer
    public private(set) var vertexBuffer: MTLBuffer?

    /// Metal index buffer
    public private(set) var indexBuffer: MTLBuffer?

    /// Number of indices to draw
    public private(set) var indexCount: Int = 0

    /// Storage for 4D vertices
    public private(set) var vertices4D: [Vertex4D] = []

    /// Storage for edges (for wireframe rendering)
    private var edges: [(Int, Int)] = []

    /// Storage for faces
    private var faces: [[Int]] = []

    /// Color map for visualization
    private var colorMap: [(position: Float, color: SIMD4<Float>)] = [
        (-1.0, SIMD4<Float>(0.0, 0.0, 0.5, 1.0)),  // Deep blue
        (-0.5, SIMD4<Float>(0.0, 0.5, 1.0, 1.0)),  // Blue
        (0.0, SIMD4<Float>(1.0, 1.0, 1.0, 1.0)),  // White
        (0.5, SIMD4<Float>(1.0, 0.5, 0.0, 1.0)),  // Orange
        (1.0, SIMD4<Float>(0.5, 0.0, 0.0, 1.0)),  // Deep red
    ]

    /// The timestamp for animation
    private var startTime: TimeInterval = Date().timeIntervalSince1970

    /// Frame counter for performance metrics
    private var frameCounter: UInt64 = 0

    /// Last frame time for FPS calculation
    private var lastFrameTime: TimeInterval = 0

    /// Current frames per second
    private var currentFPS: Double = 0

    /// Whether the renderer is initialized
    private var isInitialized: Bool = false

    /// Original vertices for 4D
    private var originalVertices4D: [SIMD4<Float>] = []

    // MARK: - Type Definitions

    /// 3D vertex for rendering
    public struct Vertex3D {
        var position: SIMD3<Float>
        var normal: SIMD3<Float>
        var color: SIMD4<Float>
        var texCoord: SIMD2<Float>

        public init(
            position: SIMD3<Float>,
            normal: SIMD3<Float>,
            color: SIMD4<Float>,
            texCoord: SIMD2<Float>
        ) {
            self.position = position
            self.normal = normal
            self.color = color
            self.texCoord = texCoord
        }
    }

    /// Vertex struct for 4D data
    public struct Vertex4D {
        var position: SIMD4<Float>
        var normal: SIMD4<Float>
        var color: SIMD4<Float>
        var texCoord: SIMD2<Float>

        public init(
            position: SIMD4<Float>,
            normal: SIMD4<Float> = SIMD4<Float>(0, 0, 0, 1),
            color: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 1),
            texCoord: SIMD2<Float> = SIMD2<Float>(0, 0)
        ) {
            self.position = position
            self.normal = normal
            self.color = color
            self.texCoord = texCoord
        }
    }

    /// Uniforms struct that matches the shader expectations
    struct Uniforms {
        var modelMatrix: simd_float4x4
        var viewMatrix: simd_float4x4
        var projectionMatrix: simd_float4x4
        var normalMatrix: simd_float3x3
        var time: Float
        var options: SIMD4<UInt32>  // Packed options (wireframe, showNormals, specialEffects, visualizationMode)
    }

    /// Defines 4D rotation planes
    enum RotationPlane4D {
        case xy, xz, xw, yz, yw, zw
    }

    // MARK: - Additional Properties

    // MARK: - Initialization

    public override init() {
        super.init()

        // Initialize with default settings
        startTime = Date().timeIntervalSinceReferenceDate
        print("GrafRenderer initialized at \(startTime)")

        // Initialize expression parser
        expressionParser = ExpressionParser()
    }

    // MARK: - Metal Setup

    /// Sets up the Metal rendering environment
    /// - Parameters:
    ///   - device: The Metal device to use
    ///   - view: The MTKView to render into
    public func setupMetal(device: MTLDevice, view: MTKView) {
        print("GrafRenderer.setupMetal called with device: \(device.name)")
        self.device = device

        // Create command queue
        commandQueue = device.makeCommandQueue()
        print("Command queue created: \(commandQueue != nil)")

        // Set up depth stencil state for proper 3D rendering
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        self.depthState = device.makeDepthStencilState(descriptor: depthDescriptor)
        print("Depth stencil state created: \(depthState != nil)")

        // Set up sampler state for texturing
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .linear
        samplerDescriptor.normalizedCoordinates = true
        self.samplerState = device.makeSamplerState(descriptor: samplerDescriptor)
        print("Sampler state created: \(samplerState != nil)")

        // Set up the rendering pipeline
        setupRenderPipeline(device: device, view: view)

        // Create initial visualization
        print("Creating initial tesseract visualization")
        generateTesseract(scale: 1.0)
        print("Generated tesseract with \(vertices4D.count) vertices and \(edges.count) edges")

        isInitialized = true
        print(
            "GrafRenderer.setupMetal completed successfully, initialization status: \(isInitialized)"
        )
    }

    /// Sets up the rendering pipeline with proper shaders and vertex descriptor
    private func setupRenderPipeline(device: MTLDevice, view: MTKView) {
        do {
            // Get default library containing Metal shaders
            guard let library = device.makeDefaultLibrary() else {
                print("ERROR: Failed to load default Metal library")
                return
            }

            print("Default Metal library loaded")

            // List available shader functions for debugging
            let functions = library.functionNames
            print("Available shader functions: \(functions)")

            // Verify shader function names match what's in your Metal file
            guard let vertexFunction = library.makeFunction(name: "vertexShader"),
                let fragmentFunction = library.makeFunction(name: "fragmentShader")
            else {
                print("ERROR: Failed to find shader functions")
                return
            }

            print("Shader functions loaded")

            // Create standard render pipeline descriptor
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.label = "Standard Pipeline"
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

            if view.depthStencilPixelFormat != .invalid {
                pipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
                print("Using depth format: \(view.depthStencilPixelFormat.rawValue)")
            } else {
                print("WARNING: No depth format specified in view")
                // Set a default depth format if not specified
                view.depthStencilPixelFormat = .depth32Float
                pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            }

            // Configure vertex descriptor to match Vertex3D structure in the shader
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

            // Texture coordinates (float2)
            vertexDescriptor.attributes[3].format = .float2
            vertexDescriptor.attributes[3].offset =
                MemoryLayout<SIMD3<Float>>.stride * 2 + MemoryLayout<SIMD4<Float>>.stride
            vertexDescriptor.attributes[3].bufferIndex = 0

            // Layout configuration
            vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex3D>.stride
            vertexDescriptor.layouts[0].stepRate = 1
            vertexDescriptor.layouts[0].stepFunction = .perVertex

            // Assign the vertex descriptor to the pipeline
            pipelineDescriptor.vertexDescriptor = vertexDescriptor

            print("Vertex descriptor configured: stride=\(MemoryLayout<Vertex3D>.stride)")

            // Create standard pipeline state
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            print("Pipeline state created successfully")

            // Create wireframe pipeline state
            let wireframeDescriptor = pipelineDescriptor.copy() as! MTLRenderPipelineDescriptor
            wireframeDescriptor.label = "Wireframe Pipeline"

            wireframePipelineState = try device.makeRenderPipelineState(
                descriptor: wireframeDescriptor)
            print("Wireframe pipeline state created successfully")

            // Create points pipeline state
            let pointsDescriptor = pipelineDescriptor.copy() as! MTLRenderPipelineDescriptor
            pointsDescriptor.label = "Points Pipeline"

            pointsPipelineState = try device.makeRenderPipelineState(descriptor: pointsDescriptor)
            print("Points pipeline state created successfully")

        } catch {
            print("ERROR: Failed to create pipeline state: \(error)")

            // Get more detailed error information
            if let nsError = error as NSError? {
                for (key, value) in nsError.userInfo {
                    print("  Error detail: \(key): \(value)")
                }
            }
        }
    }
    // MARK: - 4D Primitive Generation

    private func generateTesseract(scale: Float) {
        print("Generating tesseract with scale \(scale)")

        // Create all 16 vertices of the tesseract
        var vertices: [SIMD4<Float>] = []

        // Generate vertices for a tesseract (4D hypercube)
        for w in [0, 1] {
            for z in [0, 1] {
                for y in [0, 1] {
                    for x in [0, 1] {
                        // Map from 0,1 coordinates to -1,1 range
                        let position =
                            SIMD4<Float>(
                                Float(x) * 2 - 1,
                                Float(y) * 2 - 1,
                                Float(z) * 2 - 1,
                                Float(w) * 2 - 1
                            ) * scale

                        vertices.append(position)
                    }
                }
            }
        }

        // Generate edges for the tesseract
        edges = []

        // Connect vertices that differ by exactly one bit in their binary representation
        for i in 0..<vertices.count {
            for j in (i + 1)..<vertices.count {
                // Count differing coordinates
                var diffCount = 0
                let v1 = vertices[i]
                let v2 = vertices[j]

                if abs(v1.x - v2.x) > 0.01 { diffCount += 1 }
                if abs(v1.y - v2.y) > 0.01 { diffCount += 1 }
                if abs(v1.z - v2.z) > 0.01 { diffCount += 1 }
                if abs(v1.w - v2.w) > 0.01 { diffCount += 1 }

                // Add edge if exactly one coordinate differs
                if diffCount == 1 {
                    edges.append((i, j))
                }
            }
        }

        vertices4D = vertices.map { position -> Vertex4D in
            // Calculate normalized position for color mapping
            let lengthPos = length(position)
            let normalizedPos = position / max(lengthPos, 0.0001)
            let color = SIMD4<Float>(
                (normalizedPos.x + 1.0) * 0.5,  // Map from [-1,1] to [0,1]
                (normalizedPos.y + 1.0) * 0.5,
                (normalizedPos.z + 1.0) * 0.5,
                1.0
            )

            return Vertex4D(
                position: position,
                normal: normalizedPos,  // Normal points outward for hypersphere
                color: color,
                texCoord: SIMD2<Float>((normalizedPos.x + 1.0) * 0.5, (normalizedPos.y + 1.0) * 0.5)
            )
        }

        // Process for rendering (project 4D to 3D)
        updateBuffers()

        print(
            "Tesseract generation complete: \(vertices4D.count) vertices, \(edges.count) edges, \(faces.count) faces"
        )
    }

    private func generateHypersphere(scale: Float, resolution: Int) {
        debugMessage = "Generating hypersphere with scale \(scale) and resolution \(resolution)"
        print(debugMessage)

        var vertices: [SIMD4<Float>] = []

        // Generate vertices for a 4D hypersphere using parametric equations
        let stepTheta = 2.0 * Float.pi / Float(resolution)
        let stepPhi = Float.pi / Float(resolution)
        let stepPsi = Float.pi / Float(resolution)

        for i in 0...resolution {
            let theta = Float(i) * stepTheta

            for j in 0...resolution {
                let phi = Float(j) * stepPhi

                // Use fewer points in 4th dimension for performance
                for k in 0...(resolution / 4) {
                    let psi = Float(k) * stepPsi

                    // 4D spherical coordinates
                    let x = scale * sin(psi) * sin(phi) * cos(theta)
                    let y = scale * sin(psi) * sin(phi) * sin(theta)
                    let z = scale * sin(psi) * cos(phi)
                    let w = scale * cos(psi)

                    vertices.append(SIMD4<Float>(x, y, z, w))
                }
            }
        }

        // Limit vertices for performance
        if vertices.count > 10000 {
            print("Limiting hypersphere vertices from \(vertices.count) to 10000 for performance")
            vertices = Array(vertices.prefix(10000))
        }

        // Generate edges by connecting nearby vertices
        edges = []

        // Simple approach: connect vertices if they're within a distance threshold
        let connectionThreshold: Float = scale * 0.3

        // Track the maximum edges for vertices in total - improved efficiency
        let maxTotalEdges = vertices.count * 3

        // Create a spatial grid for faster neighbor finding
        var spatialGrid = [Int: [Int]]()  // Grid cell ID -> vertex indices
        let gridSize: Float = scale * 2.0 / connectionThreshold

        // Place vertices in the grid
        for (i, vertex) in vertices.enumerated() {
            let gx = Int((vertex.x + scale) / connectionThreshold)
            let gy = Int((vertex.y + scale) / connectionThreshold)
            let gz = Int((vertex.z + scale) / connectionThreshold)
            let gw = Int((vertex.w + scale) / connectionThreshold)

            let gridKey =
                gx + gy * Int(gridSize) + gz * Int(gridSize * gridSize) + gw
                * Int(gridSize * gridSize * gridSize)

            if spatialGrid[gridKey] == nil {
                spatialGrid[gridKey] = [i]
            } else {
                spatialGrid[gridKey]!.append(i)
            }
        }

        // Flag to break out of nested loops
        var shouldBreakOuter = false

        // Label the outer loop for breaking
        outerLoop: for i in 0..<vertices.count {
            let vertex = vertices[i]

            // Get grid cell for this vertex
            let gx = Int((vertex.x + scale) / connectionThreshold)
            let gy = Int((vertex.y + scale) / connectionThreshold)
            let gz = Int((vertex.z + scale) / connectionThreshold)
            let gw = Int((vertex.w + scale) / connectionThreshold)

            // Check this cell and adjacent cells
            for dx in -1...1 {
                for dy in -1...1 {
                    for dz in -1...1 {
                        for dw in -1...1 {
                            let nx = max(0, min(Int(gridSize) - 1, gx + dx))
                            let ny = max(0, min(Int(gridSize) - 1, gy + dy))
                            let nz = max(0, min(Int(gridSize) - 1, gz + dz))
                            let nw = max(0, min(Int(gridSize) - 1, gw + dw))

                            let neighborKey =
                                nx + ny * Int(gridSize) + nz * Int(gridSize * gridSize) + nw
                                * Int(gridSize * gridSize * gridSize)

                            if let neighbors = spatialGrid[neighborKey] {
                                // Track edges per vertex to maintain balanced connectivity
                                var edgesForVertex = 0
                                let maxEdgesPerVertex = 10  // Limit connections per vertex for better visualization

                                for j in neighbors {
                                    // Skip self-connections and duplicates by only connecting to higher indices
                                    if j <= i { continue }

                                    // Extract position vectors for cleaner distance calculation
                                    let pos1 = vertices[i]
                                    let pos2 = vertices[j]

                                    // Calculate distance using simd for efficiency
                                    let distance = simd_distance(pos1, pos2)

                                    // Connect vertices if they're close enough
                                    if distance < connectionThreshold {
                                        edges.append((i, j))
                                        edgesForVertex += 1

                                        // Limit edges per individual vertex for balanced visualization
                                        if edgesForVertex >= maxEdgesPerVertex {
                                            break
                                        }

                                        // Also check global edge limit
                                        if edges.count >= maxTotalEdges {
                                            shouldBreakOuter = true
                                            break
                                        }
                                    }
                                }

                                // Break out of outer loops if we've hit the global limit
                                if shouldBreakOuter {
                                    break outerLoop
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func generateDuocylinder(scale: Float, resolution: Int) {
        print("Generating duocylinder with scale \(scale) and resolution \(resolution)")
        debugMessage = "Generating duocylinder..."

        // IMPROVED: Limit the resolution to a reasonable value to prevent hangs
        let safeResolution = min(resolution, 24)

        // Calculate optimal step sizes for circle sampling
        let theta1Steps = safeResolution
        let r1Steps = max(3, safeResolution / 4)
        let theta2Steps = safeResolution
        let r2Steps = max(3, safeResolution / 4)

        // IMPROVED: Progress tracking for UI responsiveness
        let totalGenerationSteps = theta1Steps * r1Steps * theta2Steps * r2Steps
        var currentStep = 0

        var vertices: [SIMD4<Float>] = []
        var vertexMap = [String: Int]()  // Map position to index for quick lookup

        // Generate vertices for a duocylinder (product of two disks)
        let radius = scale * 0.5

        let stepTheta1 = 2.0 * Float.pi / Float(theta1Steps)
        let stepR1 = radius / Float(r1Steps)
        let stepTheta2 = 2.0 * Float.pi / Float(theta2Steps)
        let stepR2 = radius / Float(r2Steps)

        // Generate the duocylinder as a product of two disks
        for i in 0...theta1Steps {
            let theta1 = Float(i) * stepTheta1

            for j in 0...r1Steps {
                let r1 = Float(j) * stepR1

                for k in 0...theta2Steps {
                    let theta2 = Float(k) * stepTheta2

                    for l in 0...r2Steps {
                        let r2 = Float(l) * stepR2

                        // IMPROVED: Skip points to reduce vertex count but maintain shape
                        // Skip more points in the interior while keeping the shape boundaries
                        if j > 0 && j < r1Steps && l > 0 && l < r2Steps {
                            // Skip interior points using a pattern based on indices
                            if (i + k) % 2 != 0 || (j + l) % 2 != 0 {
                                continue
                            }
                        }

                        // Calculate position using disk parametrization
                        let x = r1 * cos(theta1)
                        let y = r1 * sin(theta1)
                        let z = r2 * cos(theta2)
                        let w = r2 * sin(theta2)

                        // IMPROVED: Create a key for deduplication
                        let key = "\(Int(x*1000)),\(Int(y*1000)),\(Int(z*1000)),\(Int(w*1000))"

                        // Skip duplicates through key lookup
                        if vertexMap[key] == nil {
                            vertices.append(SIMD4<Float>(x, y, z, w))
                            vertexMap[key] = vertices.count - 1
                        }

                        // Update progress periodically
                        currentStep += 1
                        if currentStep % 1000 == 0 {
                            let progress = Float(currentStep) / Float(totalGenerationSteps)
                            debugMessage =
                                "Generating DuoCylinder: \(Int(progress * 100))% complete"
                        }

                        // IMPROVED: Limit total vertices for performance
                        if vertices.count >= 5000 {
                            break
                        }
                    }

                    if vertices.count >= 5000 {
                        break
                    }
                }

                if vertices.count >= 5000 {
                    break
                }
            }

            if vertices.count >= 5000 {
                break
            }
        }

        print("Generated \(vertices.count) vertices for duocylinder")
        debugMessage = "Generated \(vertices.count) vertices for duocylinder"

        // IMPROVED: Efficient edge generation using spatial grid
        edges = []

        // Calculate an appropriate connection threshold
        let connectionThreshold = scale * 0.3

        // Create a spatial grid to find nearby vertices efficiently
        let gridSize = 10  // Grid divisions
        var spatialGrid = [Int: [Int]]()  // Maps grid cell to vertex indices

        // Place vertices in the grid
        for (i, vertex) in vertices.enumerated() {
            let gridX = Int((vertex.x + scale) / (2.0 * scale) * Float(gridSize))
            let gridY = Int((vertex.y + scale) / (2.0 * scale) * Float(gridSize))
            let gridZ = Int((vertex.z + scale) / (2.0 * scale) * Float(gridSize))
            let gridW = Int((vertex.w + scale) / (2.0 * scale) * Float(gridSize))

            // Create a 1D key from 4D grid coordinates
            let gridKey =
                gridX + gridY * gridSize + gridZ * gridSize * gridSize + gridW * gridSize * gridSize
                * gridSize

            if spatialGrid[gridKey] == nil {
                spatialGrid[gridKey] = [i]
            } else {
                spatialGrid[gridKey]!.append(i)
            }
        }

        // Find edges by connecting vertices in same or neighboring grid cells
        var shouldBreakOuter = false
        outerLoop: for (i, vertex) in vertices.enumerated() {
            let gridX = Int((vertex.x + scale) / (2.0 * scale) * Float(gridSize))
            let gridY = Int((vertex.y + scale) / (2.0 * scale) * Float(gridSize))
            let gridZ = Int((vertex.z + scale) / (2.0 * scale) * Float(gridSize))
            let gridW = Int((vertex.w + scale) / (2.0 * scale) * Float(gridSize))

            // Check neighboring grid cells (only check a subset of neighbors for efficiency)
            for dx in 0...1 {
                for dy in 0...1 {
                    for dz in 0...1 {
                        for dw in 0...1 {
                            let nx = min(gridX + dx, gridSize - 1)
                            let ny = min(gridY + dy, gridSize - 1)
                            let nz = min(gridZ + dz, gridSize - 1)
                            let nw = min(gridW + dw, gridSize - 1)

                            let neighborKey =
                                nx + ny * gridSize + nz * gridSize * gridSize + nw * gridSize
                                * gridSize * gridSize

                            // Connect to vertices in this cell
                            if let neighbors = spatialGrid[neighborKey] {
                                // Track edges per vertex to maintain balanced connectivity
                                var edgesForVertex = 0
                                let maxEdgesPerVertex = 10  // Limit connections per vertex for better visualization

                                for j in neighbors {
                                    // Skip self-connections and duplicates by only connecting to higher indices
                                    if j <= i { continue }

                                    // Extract position vectors for cleaner distance calculation
                                    let pos1 = vertices[i]
                                    let pos2 = vertices[j]

                                    // Calculate distance using simd for efficiency
                                    let distance = simd_distance(pos1, pos2)

                                    // Connect vertices if they're close enough
                                    if distance < connectionThreshold {
                                        edges.append((i, j))
                                        edgesForVertex += 1

                                        // Limit edges per individual vertex for balanced visualization
                                        if edgesForVertex >= maxEdgesPerVertex {
                                            break
                                        }

                                        // Also check global edge limit
                                        if edges.count >= vertices.count * 3 {
                                            shouldBreakOuter = true
                                            break
                                        }
                                    }
                                }

                                // Break out of outer loops if we've hit the global limit
                                if shouldBreakOuter {
                                    break outerLoop
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func generateCustomFunction(expression: String, scale: Float, resolution: Int) {
        print(
            "Generating custom function with expression: \(expression), scale: \(scale), resolution: \(resolution)"
        )

        // Store the expression for future reference
        customFunctionExpression = expression

        // IMPROVED: Check if the expression is valid with proper error handling
        guard expressionParser.isValid(expression: expression) else {
            print("Error: Invalid expression: \(expression)")
            // Provide user feedback via debugMessage
            debugMessage = "Error: Invalid expression syntax in \"\(expression)\""

            // Use a simple default function instead of crashing
            customFunctionExpression = "sin(x) * cos(y)"
            return generateCustomFunction(
                expression: customFunctionExpression, scale: scale, resolution: min(16, resolution))
        }

        // IMPROVED: Limit resolution for complex custom functions to prevent hangs
        let safeResolution = min(resolution, 24)  // Cap at 24 for safety

        var vertices: [SIMD4<Float>] = []

        // IMPROVED: Use a more efficient sampling strategy with dynamic resolution
        // This adaptive approach uses higher resolution for simpler expressions
        let stepCount = safeResolution
        let step = 2.0 / Float(max(3, stepCount - 1)) * scale

        // IMPROVED: Create a progress reporting mechanism
        var progress: Float = 0.0
        let totalPoints = stepCount * stepCount * stepCount
        var pointsProcessed = 0

        // Sample 3D space (x,y,z) and compute w as function result
        for xi in 0..<stepCount {
            let x = -scale + Float(xi) * step

            for yi in 0..<stepCount {
                let y = -scale + Float(yi) * step

                // IMPROVED: Report progress periodically
                if yi % 5 == 0 {
                    progress = Float(pointsProcessed) / Float(totalPoints)
                    debugMessage = "Generating custom function: \(Int(progress * 100))% complete"
                }

                for zi in 0..<stepCount {
                    let z = -scale + Float(zi) * step
                    pointsProcessed += 1

                    // Normalize coordinates to stay within reasonable bounds
                    let nx = x / scale
                    let ny = y / scale
                    let nz = z / scale

                    // Evaluate the function with improved error handling
                    guard
                        let value = expressionParser.evaluate(
                            expression: expression, x: Double(nx), y: Double(ny), t: Double(nz)
                        )
                    else {
                        continue  // Skip this point if evaluation fails
                    }

                    // Skip NaN or infinite results
                    if !value.isFinite {
                        continue
                    }

                    // Use the function value as the w-coordinate
                    let position = SIMD4<Float>(nx, ny, nz, Float(value) * 0.5) * scale
                    vertices.append(position)

                    // IMPROVED: Limit total vertices to prevent performance issues
                    if vertices.count >= 5000 {
                        break
                    }
                }

                // Break early if we already have enough vertices
                if vertices.count >= 5000 {
                    break
                }
            }

            // Break early if we already have enough vertices
            if vertices.count >= 5000 {
                break
            }
        }

        // Report completion
        debugMessage = "Custom function generation complete: \(vertices.count) vertices"

        // IMPROVED: Handle the case where no valid vertices were generated
        if vertices.isEmpty {
            debugMessage = "Error: Function evaluation produced no valid points"
            // Create a simple placeholder object
            vertices.append(SIMD4<Float>(0, 0, 0, 0))
            vertices.append(SIMD4<Float>(scale, 0, 0, 0))
            vertices.append(SIMD4<Float>(0, scale, 0, 0))
            vertices.append(SIMD4<Float>(0, 0, scale, 0))
        }

        // IMPROVED: Create edges using a more efficient algorithm for better performance
        edges = []
        // Use a fixed connection distance based on point density
        let connectionThreshold = scale * 1.5 / Float(stepCount)

        // Use a spatial grid to improve edge finding performance
        let gridSize = Int(ceil(2.0 * scale / connectionThreshold))
        var spatialGrid = [Int: [Int]]()  // Maps grid cell to vertex indices

        // Place vertices in the grid
        for (i, vertex) in vertices.enumerated() {
            let gridX = Int((vertex.x + scale) / connectionThreshold)
            let gridY = Int((vertex.y + scale) / connectionThreshold)
            let gridZ = Int((vertex.z + scale) / connectionThreshold)
            let gridKey = gridX + gridY * gridSize + gridZ * gridSize * gridSize

            if spatialGrid[gridKey] == nil {
                spatialGrid[gridKey] = [i]
            } else {
                spatialGrid[gridKey]!.append(i)
            }
        }

        // Find nearby vertices using the grid
        for (i, vertex) in vertices.enumerated() {
            let gridX = Int((vertex.x + scale) / connectionThreshold)
            let gridY = Int((vertex.y + scale) / connectionThreshold)
            let gridZ = Int((vertex.z + scale) / connectionThreshold)

            // Check neighboring grid cells
            for dx in -1...1 {
                for dy in -1...1 {
                    for dz in -1...1 {
                        let nx = gridX + dx
                        let ny = gridY + dy
                        let nz = gridZ + dz

                        // Skip if out of bounds
                        if nx < 0 || ny < 0 || nz < 0 || nx >= gridSize || ny >= gridSize
                            || nz >= gridSize
                        {
                            continue
                        }

                        let neighborKey = nx + ny * gridSize + nz * gridSize * gridSize

                        if let neighbors = spatialGrid[neighborKey] {
                            // Track edges per vertex to maintain balanced connectivity
                            var edgesForVertex = 0
                            let maxEdgesPerVertex = 10  // Limit connections per vertex for better visualization

                            for j in neighbors {
                                // Skip self-connections and duplicates by only connecting to higher indices
                                if j <= i { continue }

                                // Extract position vectors for cleaner distance calculation
                                let pos1 = vertex
                                let pos2 = vertices[j]

                                // Calculate distance using simd for efficiency
                                let distance = simd_distance(pos1, pos2)

                                // Connect vertices if they're close enough
                                if distance < connectionThreshold {
                                    edges.append((i, j))
                                    edgesForVertex += 1

                                    // Limit edges per individual vertex for balanced visualization
                                    if edgesForVertex >= maxEdgesPerVertex {
                                        break
                                    }

                                    // Also check global edge limit
                                    if edges.count >= vertices.count * 3 {
                                        break
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func generateCliffordTorus(scale: Float, resolution: Int) {
        print("Generating Clifford torus with scale \(scale) and resolution \(resolution)")
        let vertices: [SIMD4<Float>] = []
        vertices4D = vertices.map { position in
            Vertex4D(
                position: position,
                normal: normalize(position),
                color: SIMD4<Float>(1, 1, 1, 1)
            )
        }
        updateBuffers()
    }

    private func generateQuaternionVisualization(scale: Float, resolution: Int) {
        print(
            "Generating quaternion visualization with scale \(scale) and resolution \(resolution)")
        let vertices: [SIMD4<Float>] = []
        vertices4D = vertices.map { position in
            Vertex4D(
                position: position,
                normal: normalize(position),
                color: SIMD4<Float>(1, 1, 1, 1)
            )
        }
        updateBuffers()
    }

    func setCustomFunction(_ expression: String, scale: Float, resolution: Int) {
        generateCustomFunction(expression: expression, scale: scale, resolution: resolution)
    }

    private func createRotationMatrix4D(plane: RotationPlane4D, angle: Float) -> simd_float4x4 {
        return simd_float4x4()
    }

    private func createRotationMatrices() -> [simd_float4x4] {
        return []
    }

    private func rotateVertices4D() -> [Vertex4D] {
        return []
    }

    private func project4DTo3D() -> [Vertex3D] {
        return []
    }

    public func updateBuffers() {
        print(
            "GrafRenderer.updateBuffers called with \(vertices4D.count) vertices and \(edges.count) edges"
        )

        // Create vertex buffer if needed
        if vertices4D.count > 0 {
            let vertexDataSize = vertices4D.count * MemoryLayout<Vertex4D>.stride
            vertexBuffer = device?.makeBuffer(
                bytes: vertices4D, length: vertexDataSize, options: .storageModeShared)

            if vertexBuffer == nil {
                print("Failed to create vertex buffer")
            } else {
                print("Created vertex buffer with \(vertices4D.count) vertices")
            }
        }

        // Convert edges to indices if needed
        var indices = [UInt32]()
        for edge in edges {
            indices.append(UInt32(edge.0))
            indices.append(UInt32(edge.1))
        }

        // Create index buffer if needed
        if indices.count > 0 {
            let indexDataSize = indices.count * MemoryLayout<UInt32>.stride
            indexBuffer = device?.makeBuffer(
                bytes: indices, length: indexDataSize, options: .storageModeShared)
            indexCount = indices.count

            if indexBuffer == nil {
                print("Failed to create index buffer")
            } else {
                print("Created index buffer with \(indices.count) indices")
            }
        }

        print(
            "Buffers updated: vertexBuffer=\(vertexBuffer != nil), indexBuffer=\(indexBuffer != nil), indexCount=\(indexCount)"
        )
    }

    private func updateCurrentVisualization() {
        // Check if renderer is initialized
        guard isInitialized else {
            return
        }

        // If we don't have any 4D vertices, nothing to update
        if vertices4D.isEmpty {
            return
        }

        // Store original positions separately from the current transformed vertices
        // This is crucial - we need to maintain original 4D coordinates to apply rotations correctly
        if originalVertices4D.isEmpty && !vertices4D.isEmpty {
            // First-time initialization of original vertices
            originalVertices4D = vertices4D.map { $0.position }
        }

        // Extract the original 4D vertex positions to apply rotations to
        let positionsToTransform =
            originalVertices4D.isEmpty ? vertices4D.map { $0.position } : originalVertices4D

        // Set up the rotation angles
        let rotations = (
            xy: rotationAngles.xy,
            xz: rotationAngles.xz,
            xw: rotationAngles.xw,
            yz: rotationAngles.yz,
            yw: rotationAngles.yw,
            zw: rotationAngles.zw
        )

        // Convert projection type
        let projType: GA4D.GA4DMetalBridge.ProjectionType
        switch projectionType {
        case .stereographic: projType = .stereographic
        case .perspective: projType = .perspective
        case .orthographic: projType = .orthographic
        }

        // Transform vertices using GA4D
        let transformed3D = GA4D.GA4DMetalBridge.transformVertices(
            vertices4D: positionsToTransform,
            rotations: rotations,
            projectionType: projType
        )

        // Ensure arrays have same size for consistency
        guard transformed3D.count == positionsToTransform.count else {
            print(
                "Warning: Transformation resulted in different vertex count. Expected: \(positionsToTransform.count), Got: \(transformed3D.count)"
            )
            return
        }

        // Update vertex data with transformed positions
        for i in 0..<min(vertices4D.count, transformed3D.count) {
            // Update vertex positions with the 3D transformed positions
            // Keep the original vertex normals, colors, and texture coordinates
            vertices4D[i] = Vertex4D(
                position: SIMD4<Float>(
                    transformed3D[i].x, transformed3D[i].y, transformed3D[i].z, 0),
                normal: vertices4D[i].normal,
                color: vertices4D[i].color,
                texCoord: vertices4D[i].texCoord
            )
        }

        // Regenerate buffers with updated vertex data
        updateBuffers()
    }

    private func colorFromMap(_ value: Float) -> SIMD4<Float> {
        return SIMD4<Float>(1, 1, 1, 1)
    }

    func setColorMap(_ colorMap: [(position: Float, color: SIMD4<Float>)]) {
        self.colorMap = colorMap.sorted { $0.position < $1.position }
        updateCurrentVisualization()
    }

    func setRotation(plane: String, angle: Float) {
        // Update the appropriate rotation angle based on the plane
        switch plane.lowercased() {
        case "xy":
            rotationAngles.xy = angle
        case "xz":
            rotationAngles.xz = angle
        case "xw":
            rotationAngles.xw = angle
        case "yz":
            rotationAngles.yz = angle
        case "yw":
            rotationAngles.yw = angle
        case "zw":
            rotationAngles.zw = angle
        default:
            print("Unknown rotation plane: \(plane)")
            return
        }

        // After updating rotation angles, reapply transformations to current visualization
        updateCurrentVisualization()
    }

    func rotate4D(plane: String, delta: Float) {
        // Increment the appropriate rotation angle based on the plane
        switch plane.lowercased() {
        case "xy":
            rotationAngles.xy += delta
            // Normalize to [0, 2π]
            rotationAngles.xy = fmodf(rotationAngles.xy, Float.pi * 2)
        case "xz":
            rotationAngles.xz += delta
            rotationAngles.xz = fmodf(rotationAngles.xz, Float.pi * 2)
        case "xw":
            rotationAngles.xw += delta
            rotationAngles.xw = fmodf(rotationAngles.xw, Float.pi * 2)
        case "yz":
            rotationAngles.yz += delta
            rotationAngles.yz = fmodf(rotationAngles.yz, Float.pi * 2)
        case "yw":
            rotationAngles.yw += delta
            rotationAngles.yw = fmodf(rotationAngles.yw, Float.pi * 2)
        case "zw":
            rotationAngles.zw += delta
            rotationAngles.zw = fmodf(rotationAngles.zw, Float.pi * 2)
        default:
            print("Unknown rotation plane: \(plane)")
            return
        }

        // After updating rotation angles, reapply transformations to current visualization
        updateCurrentVisualization()
    }

    func handleMouseDrag(dx: Float, dy: Float) {
        switch interactionMode {
        case .rotate:
            // 3D camera rotation
            rotate(dx: dx * 0.01, dy: dy * 0.01)
        case .pan:
            // 3D camera panning
            pan(dx: dx * 0.01, dy: dy * 0.01)
        case .select:
            // In select mode, convert to appropriate 4D rotation based on mouse position
            // For example, map horizontal movement to XY rotation and vertical to XZ rotation
            rotate4D(plane: "xy", delta: dx * 0.01)
            rotate4D(plane: "xz", delta: dy * 0.01)
        case .sketch:
            // Sketch mode not implemented yet
            break
        }
    }

    func resetCamera() {
        camera = Camera()
    }

    func setVisualizationType(_ type: VisualizationType) {
        currentVisType = type
        updateCurrentVisualization()
    }

    func getCamera() -> Camera {
        return camera
    }

    func setProjectionType(_ type: ProjectionType) {
        self.projectionType = type
    }

    func updateProjectionMatrix() {
        // ... existing updateProjectionMatrix implementation ...
    }

    func visualizationTypeToIndex(_ type: VisualizationType) -> UInt32 {
        // ... existing visualizationTypeToIndex implementation ...
        return 0
    }

    func handleProjectionTypeChanged(index: Int) {
        // ... existing handleProjectionTypeChanged implementation ...
    }

    public func updateVisualization(data: Graf.VisualizationData) {
        print(
            "GrafRenderer.updateVisualization called with \(data.vertices.count) vertices and \(data.edges.count) edges"
        )

        // Store original 4D vertices
        var newVertices4D = [Vertex4D]()
        for vertex in data.originalVertices4D {
            newVertices4D.append(
                Vertex4D(
                    position: vertex,
                    normal: SIMD4<Float>(0, 0, 0, 1),
                    color: SIMD4<Float>(1, 1, 1, 1)
                ))
        }
        vertices4D = newVertices4D

        // Store edges and faces
        edges = data.edges

        // Update buffers with new data
        updateBuffers()

        print("Visualization updated with \(vertices4D.count) vertices and \(edges.count) edges")
        print("Calling updateBuffers...")
        updateBuffers()
    }

    func handleVisualizationUpdate(typeIndex: Int, resolution: Int, scale: Float) {
        // ... existing handleVisualizationUpdate implementation ...
    }

    // MARK: - Camera Control Methods
    func rotate(dx: Float, dy: Float) {
        camera.rotateX += dy
        camera.rotateY += dx
    }

    func pan(dx: Float, dy: Float) {
        camera.panX += dx
        camera.panY += dy
    }

    func zoom(factor: Float) {
        camera.zoom *= factor
        camera.zoom = max(0.1, min(camera.zoom, 10.0))
    }

    /// Handles mouse click events
    /// - Parameters:
    ///   - location: The location of the click in the view
    ///   - size: The size of the view
    ///   - view: The MTKView that received the click
    func handleMouseClick(at location: CGPoint, size: CGSize, view: MTKView) {
        // Store the click location for potential selection or other click-based interactions
        let normalizedX = Float(location.x / size.width)
        let normalizedY = Float(location.y / size.height)

        print("Click detected at normalized coordinates: (\(normalizedX), \(normalizedY))")

        // TODO: Implement selection or other click-based interactions
    }

    private func updateUniformBuffers() {
        // Create model-view-projection matrix
        let currentTime = Date().timeIntervalSinceReferenceDate
        let elapsedTime = Float(currentTime - startTime) * animationSpeed

        let modelMatrix = matrix_float4x4(scaling: SIMD3<Float>(1, 1, 1))
        let viewMatrix = camera.viewMatrix()
        let projectionMatrix = camera.projectionMatrix(type: projectionType)

        // Calculate normal matrix (inverse transpose of the model-view matrix)
        let modelViewMatrix = matrix_multiply(viewMatrix, modelMatrix)
        var normalMatrix = simd_float3x3(
            SIMD3<Float>(
                modelViewMatrix.columns.0.x, modelViewMatrix.columns.0.y,
                modelViewMatrix.columns.0.z),
            SIMD3<Float>(
                modelViewMatrix.columns.1.x, modelViewMatrix.columns.1.y,
                modelViewMatrix.columns.1.z),
            SIMD3<Float>(
                modelViewMatrix.columns.2.x, modelViewMatrix.columns.2.y,
                modelViewMatrix.columns.2.z)
        )
        normalMatrix = simd_transpose(simd_inverse(normalMatrix))

        // Create uniforms
        var uniforms = Uniforms(
            modelMatrix: modelMatrix,
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            normalMatrix: normalMatrix,
            time: elapsedTime,
            options: SIMD4<UInt32>(
                wireframe ? 1 : 0, showNormals ? 1 : 0, 0, visualizationTypeToIndex(currentVisType))
        )

        // Create or update uniform buffer - safe unwrap device
        let uniformSize = MemoryLayout<Uniforms>.stride
        if uniformBuffer == nil, let device = device {
            uniformBuffer = device.makeBuffer(length: uniformSize, options: .storageModeShared)
        }

        if let buffer = uniformBuffer {
            memcpy(buffer.contents(), &uniforms, uniformSize)
        }
    }
}

// MARK: - MTKViewDelegate
extension GrafRenderer {
    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        print("View size changed to: \(size)")
        let aspect = Float(size.width / size.height)
        camera.aspectRatio = aspect
        updateProjectionMatrix()
    }

    public func draw(in view: MTKView) {
        guard isInitialized else {
            return
        }

        // Get the current drawable and command buffer
        guard let drawable = view.currentDrawable,
            let renderPassDescriptor = view.currentRenderPassDescriptor,
            let commandBuffer = commandQueue?.makeCommandBuffer()
        else {
            return
        }

        // Handle auto-rotation if enabled
        if autoRotate {
            // Get current time for animation
            let currentTime = Date().timeIntervalSinceReferenceDate
            let elapsedTime = Float(currentTime - startTime) * animationSpeed

            // Calculate time since last update to limit update frequency
            let timeSinceLastUpdate = currentTime - lastUpdateTime

            // Only update rotation angles and visualization if enough time has passed (e.g., 1/60th of a second)
            if timeSinceLastUpdate > 0.016 {  // ~60 FPS
                // Apply rotation to different planes
                rotationAngles.xy = fmodf(elapsedTime * 0.3, Float.pi * 2)
                rotationAngles.xw = fmodf(elapsedTime * 0.2, Float.pi * 2)
                rotationAngles.yw = fmodf(elapsedTime * 0.15, Float.pi * 2)

                // Update visualization with new rotation angles
                updateCurrentVisualization()

                // Update timestamp
                lastUpdateTime = currentTime
            }
        }

        // Set up render pass descriptor
        renderPassDescriptor.colorAttachments[0].clearColor = view.clearColor
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store

        // Create render encoder
        guard
            let renderEncoder = commandBuffer.makeRenderCommandEncoder(
                descriptor: renderPassDescriptor)
        else {
            return
        }

        // Set render pipeline state based on rendering mode
        if wireframe {
            renderEncoder.setRenderPipelineState(wireframePipelineState ?? pipelineState!)
        } else {
            renderEncoder.setRenderPipelineState(pipelineState!)
        }

        // Set depth state
        if let depthState = depthState {
            renderEncoder.setDepthStencilState(depthState)
        }

        // Set vertex and fragment buffers
        if let vertexBuffer = vertexBuffer {
            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        }

        // Update and set uniform buffers with current transformation matrices
        updateUniformBuffers()
        if let uniformBuffer = uniformBuffer {
            renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)
        }

        // Draw geometry
        if let indexBuffer = indexBuffer, indexCount > 0 {
            renderEncoder.drawIndexedPrimitives(
                type: .line,
                indexCount: indexCount,
                indexType: .uint32,
                indexBuffer: indexBuffer,
                indexBufferOffset: 0
            )
        } else if vertices4D.count > 0 {
            renderEncoder.drawPrimitives(
                type: .point,
                vertexStart: 0,
                vertexCount: vertices4D.count
            )
        }

        // End encoding and commit command buffer
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
