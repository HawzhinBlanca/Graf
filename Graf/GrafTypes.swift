import Foundation
import Metal
import MetalKit
import simd

// MARK: - Shared Types for Graf
// This file is the canonical source for all shared type definitions

// Create a namespace for Graf types to match how they're used in other files
public enum Graf {
    // MARK: - Visualization Types
    public enum VisualizationType: String, CaseIterable, Identifiable {
        case vertices = "Vertices"
        case wireframe = "Wireframe"
        case solid = "Solid"
        // Additional 4D visualization types
        case tesseract = "Tesseract"
        case hypersphere = "Hypersphere"
        case duocylinder = "Duocylinder"
        case cliffordTorus = "Clifford Torus"
        case quaternion = "Quaternion"
        case customFunction = "Custom Function"

        public var id: String { self.rawValue }
    }

    /// Represents different types of projections from 4D to 3D
    public enum ProjectionType: String, CaseIterable, Identifiable {
        /// Stereographic projection - preserves angles and circles
        case stereographic = "Stereographic"

        /// Perspective projection - simulates depth with distance attenuation
        case perspective = "Perspective"

        /// Orthographic projection - simply drops the 4th coordinate
        case orthographic = "Orthographic"

        public var id: String { self.rawValue }

        /// Get a description of the projection type
        public var description: String {
            switch self {
            case .stereographic:
                return "Projects from 4D to 3D using stereographic projection, preserving angles"
            case .perspective:
                return
                    "Projects from 4D to 3D with perspective, making distant objects appear smaller"
            case .orthographic:
                return "Projects from 4D to 3D by simply dropping the 4th coordinate"
            }
        }

        /// Get a projection matrix for this projection type
        public func getProjectionMatrix(
            fov: Float = Float.pi / 4, aspectRatio: Float = 1.0,
            nearZ: Float = 0.1, farZ: Float = 100.0
        ) -> matrix_float4x4 {
            switch self {
            case .stereographic:
                // Stereographic projection matrix optimized for 4D to 3D projection
                let y = 1.0 / tan(fov * 0.5)
                let x = y / aspectRatio
                let z = farZ / (farZ - nearZ)
                let w = -z * nearZ

                return matrix_float4x4(
                    SIMD4<Float>(x, 0, 0, 0),
                    SIMD4<Float>(0, y, 0, 0),
                    SIMD4<Float>(0, 0, z, 1),
                    SIMD4<Float>(0, 0, w, 0)
                )

            case .perspective:
                // Standard perspective projection matrix
                return matrix_float4x4(
                    perspectiveFov: fov,
                    aspectRatio: aspectRatio,
                    nearZ: nearZ,
                    farZ: farZ
                )

            case .orthographic:
                // Orthographic projection matrix with reasonable bounds
                let left: Float = -5
                let right: Float = 5
                let bottom: Float = -5
                let top: Float = 5

                return matrix_float4x4(
                    orthographic: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    nearZ: nearZ,
                    farZ: farZ
                )
            }
        }

        /// Project a 4D point to 3D using this projection type
        public func project4DTo3D(_ point4D: SIMD4<Float>) -> SIMD3<Float> {
            switch self {
            case .stereographic:
                // Stereographic projection formula
                let factor: Float = 1.0 / (1.0 - point4D.w * 0.1)
                return SIMD3<Float>(point4D.x, point4D.y, point4D.z) * factor

            case .perspective:
                // Perspective projection formula
                let viewDistance: Float = 5.0
                let factor = viewDistance / (viewDistance + point4D.w)
                return SIMD3<Float>(point4D.x, point4D.y, point4D.z) * factor

            case .orthographic:
                // Orthographic projection (simply drop w component)
                return SIMD3<Float>(point4D.x, point4D.y, point4D.z)
            }
        }

        /// Project a list of 4D points to 3D using this projection type
        public func projectVertices(_ vertices: [SIMD4<Float>]) -> [SIMD3<Float>] {
            return vertices.map { project4DTo3D($0) }
        }
    }

    // MARK: - Visualization Data
    public struct VisualizationData {
        // Original properties
        public var vertices: [SIMD3<Float>]
        public var normals: [SIMD3<Float>]
        public var colors: [SIMD4<Float>]
        public var indices: [UInt32]

        // New properties to match GA4DVisualizer.VisualizationData
        public var vertices3D: [SIMD3<Float>] {
            get { return vertices }
            set { vertices = newValue }
        }
        public var edges: [(Int, Int)] = []
        public var faces: [[Int]] = []
        public var originalVertices4D: [SIMD4<Float>] = []

        public init(
            vertices: [SIMD3<Float>] = [], normals: [SIMD3<Float>] = [],
            colors: [SIMD4<Float>] = [], indices: [UInt32] = [],
            edges: [(Int, Int)] = [], faces: [[Int]] = [], originalVertices4D: [SIMD4<Float>] = []
        ) {
            self.vertices = vertices
            self.normals = normals
            self.colors = colors
            self.indices = indices
            self.edges = edges
            self.faces = faces
            self.originalVertices4D = originalVertices4D
        }
    }
}

// MARK: - Extension for ProjectionType
// Moving this to file scope to fix "declaration is only valid at file scope" error
extension Graf.ProjectionType {
    /// Improved implementation of projection matrix creation
    public func improvedProjectionMatrix(
        fov: Float = Float.pi / 4, aspectRatio: Float = 1.0,
        nearZ: Float = 0.1, farZ: Float = 100.0
    ) -> matrix_float4x4 {
        switch self {
        case .stereographic:
            // Stereographic projection from 4D to 3D
            // Projects points on the 4D unit sphere to 3D space
            let scale: Float = 1.0 / tan(fov * 0.5)
            let x = scale / aspectRatio
            let y = scale

            return matrix_float4x4(
                SIMD4<Float>(x, 0, 0, 0),
                SIMD4<Float>(0, y, 0, 0),
                SIMD4<Float>(0, 0, farZ / (farZ - nearZ), 1),
                SIMD4<Float>(0, 0, -(farZ * nearZ) / (farZ - nearZ), 0)
            )

        case .perspective:
            // Standard perspective projection with modifications for 4D perspective
            let y = 1.0 / tan(fov * 0.5)
            let x = y / aspectRatio
            let z = farZ / (farZ - nearZ)
            let w = -nearZ * z

            return matrix_float4x4(
                SIMD4<Float>(x, 0, 0, 0),
                SIMD4<Float>(0, y, 0, 0),
                SIMD4<Float>(0, 0, z, 1),
                SIMD4<Float>(0, 0, w, 0)
            )

        case .orthographic:
            // Orthographic projection for 4D objects
            // Projects by simply dropping the w-coordinate
            let width: Float = 2.0 * tan(fov * 0.5)
            let height = width / aspectRatio

            return matrix_float4x4(
                SIMD4<Float>(2.0 / width, 0, 0, 0),
                SIMD4<Float>(0, 2.0 / height, 0, 0),
                SIMD4<Float>(0, 0, 1.0 / (farZ - nearZ), 0),
                SIMD4<Float>(0, 0, -nearZ / (farZ - nearZ), 1)
            )
        }
    }

    /// Improved implementation of 4D to 3D projection
    public func improvedProject(point4D: SIMD4<Float>, viewDistance: Float = 5.0) -> SIMD3<Float> {
        switch self {
        case .stereographic:
            // Stereographic projection from "north pole" (0,0,0,1)
            // Formula: (x,y,z)/(1-w)
            let denom = max(1.0 - point4D.w * 0.2, 0.1)  // Avoid division by zero
            return SIMD3<Float>(
                point4D.x / denom,
                point4D.y / denom,
                point4D.z / denom
            )

        case .perspective:
            // 4D perspective projection
            // Similar to 3D perspective but using w for depth
            let denom = max(viewDistance - point4D.w, 0.1)  // Avoid division by zero
            return SIMD3<Float>(
                point4D.x * viewDistance / denom,
                point4D.y * viewDistance / denom,
                point4D.z * viewDistance / denom
            )

        case .orthographic:
            // Simple orthographic projection - just drop the w coordinate
            return SIMD3<Float>(point4D.x, point4D.y, point4D.z)
        }
    }
}

// MARK: - Camera
public class Camera {
    public var position: SIMD3<Float>
    public var rotation: SIMD3<Float>
    public var fov: Float
    public var aspectRatio: Float
    public var nearZ: Float
    public var farZ: Float

    // Compatibility properties
    public var rotateX: Float {
        get { rotation.x }
        set { rotation.x = newValue }
    }
    public var rotateY: Float {
        get { rotation.y }
        set { rotation.y = newValue }
    }
    public var panX: Float = 0
    public var panY: Float = 0
    public var zoom: Float = 1.0

    public init(
        position: SIMD3<Float> = SIMD3<Float>(0, 0, -5),
        rotation: SIMD3<Float> = SIMD3<Float>(0, 0, 0),
        fov: Float = 45.0 * (Float.pi / 180.0),
        aspectRatio: Float = 1.0,
        nearZ: Float = 0.1,
        farZ: Float = 100.0
    ) {
        self.position = position
        self.rotation = rotation
        self.fov = fov
        self.aspectRatio = aspectRatio
        self.nearZ = nearZ
        self.farZ = farZ
    }

    public func viewMatrix() -> matrix_float4x4 {
        let translationMatrix = matrix_float4x4(translation: -position)
        let rotationMatrix = matrix_float4x4(rotation: rotation)
        return matrix_multiply(rotationMatrix, translationMatrix)
    }

    public func projectionMatrix(type: Graf.ProjectionType) -> matrix_float4x4 {
        switch type {
        case .perspective:
            return matrix_float4x4(
                perspectiveFov: fov, aspectRatio: aspectRatio, nearZ: nearZ, farZ: farZ)
        case .orthographic:
            let left: Float = -5
            let right: Float = 5
            let bottom: Float = -5
            let top: Float = 5

            let width = right - left
            let height = top - bottom
            let depth = farZ - nearZ

            return matrix_float4x4(
                SIMD4<Float>(2 / width, 0, 0, 0),
                SIMD4<Float>(0, 2 / height, 0, 0),
                SIMD4<Float>(0, 0, -2 / depth, 0),
                SIMD4<Float>(
                    -(right + left) / width, -(top + bottom) / height, -(farZ + nearZ) / depth, 1)
            )
        case .stereographic:
            // Similar to perspective, but optimized for 4D to 3D projection
            let y = 1.0 / tan(fov * 0.5)
            let x = y / aspectRatio
            let z = farZ / (farZ - nearZ)
            let w = -z * nearZ

            return matrix_float4x4(
                SIMD4<Float>(x, 0, 0, 0),
                SIMD4<Float>(0, y, 0, 0),
                SIMD4<Float>(0, 0, z, 1),
                SIMD4<Float>(0, 0, w, 0)
            )
        }
    }

    // MARK: - 4D Projection Methods

    /// Create a proper 4D projection matrix based on current settings
    public func improvedProjectionMatrix(type: Graf.ProjectionType) -> matrix_float4x4 {
        let adjustedFov = fov * zoom  // Apply camera zoom to field of view

        // Use the improved projection matrix implementation
        return type.improvedProjectionMatrix(
            fov: adjustedFov,
            aspectRatio: aspectRatio,
            nearZ: nearZ,
            farZ: farZ
        )
    }

    /// Generate a proper combined view-projection matrix for 4D rendering
    public func viewProjectionMatrix(type: Graf.ProjectionType) -> matrix_float4x4 {
        let view = viewMatrix()
        let projection = improvedProjectionMatrix(type: type)
        return matrix_multiply(projection, view)
    }
}

// MARK: - Rendering Protocol
public protocol GrafRendering: MTKViewDelegate {
    var device: MTLDevice? { get }
    var commandQueue: MTLCommandQueue? { get }
    var pipeline: MTLRenderPipelineState? { get }

    func setupMetal(device: MTLDevice, view: MTKView)
    func updateVisualization(data: Graf.VisualizationData)
    func setProjectionType(_ projectionType: Graf.ProjectionType)
    func getCamera() -> Camera
    func rotate(dx: Float, dy: Float)
    func pan(dx: Float, dy: Float)
    func zoom(factor: Float)
    func resetCamera()
}

// MARK: - Matrix Extensions
extension matrix_float4x4 {
    init(scaling: SIMD3<Float>) {
        self.init(
            SIMD4<Float>(scaling.x, 0, 0, 0),
            SIMD4<Float>(0, scaling.y, 0, 0),
            SIMD4<Float>(0, 0, scaling.z, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
    }
    init(translation: SIMD3<Float>) {
        self.init(
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(translation.x, translation.y, translation.z, 1)
        )
    }

    init(rotation: SIMD3<Float>) {
        let xMatrix = matrix_float4x4(
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, cos(rotation.x), sin(rotation.x), 0),
            SIMD4<Float>(0, -sin(rotation.x), cos(rotation.x), 0),
            SIMD4<Float>(0, 0, 0, 1))

        let yMatrix = matrix_float4x4(
            SIMD4<Float>(cos(rotation.y), 0, -sin(rotation.y), 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(sin(rotation.y), 0, cos(rotation.y), 0),
            SIMD4<Float>(0, 0, 0, 1))

        let zMatrix = matrix_float4x4(
            SIMD4<Float>(cos(rotation.z), sin(rotation.z), 0, 0),
            SIMD4<Float>(-sin(rotation.z), cos(rotation.z), 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, 0, 1))

        self = matrix_multiply(matrix_multiply(xMatrix, yMatrix), zMatrix)
    }

    init(perspectiveFov fov: Float, aspectRatio: Float, nearZ: Float, farZ: Float) {
        let yScale = 1 / tan(fov * 0.5)
        let xScale = yScale / aspectRatio
        let zRange = farZ - nearZ
        let zScale = -(farZ + nearZ) / zRange
        let wzScale = -2 * farZ * nearZ / zRange

        self.init(
            SIMD4<Float>(xScale, 0, 0, 0),
            SIMD4<Float>(0, yScale, 0, 0),
            SIMD4<Float>(0, 0, zScale, -1),
            SIMD4<Float>(0, 0, wzScale, 0)
        )
    }

    init(
        orthographic left: Float, right: Float, bottom: Float, top: Float, nearZ: Float, farZ: Float
    ) {
        let width = right - left
        let height = top - bottom
        let depth = farZ - nearZ

        self.init(
            SIMD4<Float>(2 / width, 0, 0, 0),
            SIMD4<Float>(0, 2 / height, 0, 0),
            SIMD4<Float>(0, 0, -2 / depth, 0),
            SIMD4<Float>(
                -(right + left) / width, -(top + bottom) / height, -(farZ + nearZ) / depth, 1))
    }
}

// For backward compatibility, define type aliases for existing code
public typealias VisualizationType = Graf.VisualizationType
public typealias ProjectionType = Graf.ProjectionType

// No need to redefine RenderVertex, just add a typealias if needed
// public typealias RenderVertex = GA4DVisualizer.RenderVertex
