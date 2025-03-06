//
//  GA4DVisualizer.swift
//  Graf
//
//  Created on 14/03/2025
//  Enhanced visualization system for 4D objects
//

import Foundation
import Metal
import simd

/// Enhanced visualization system for 4D objects
public class GA4DVisualizer {

    // MARK: - Types

    /// Colorization method for 4D objects
    public enum ColorMethod: Hashable, Equatable {
        case wCoordinate  // Color based on 4D w-coordinate
        case distance  // Color based on distance from origin
        case normal  // Color based on normal direction
        case curvature  // Color based on local curvature
        case cell  // Color based on cell/face ID
        case custom((SIMD4<Float>) -> SIMD4<Float>)  // Custom coloring function

        // Implement Equatable for custom case with function
        public static func == (lhs: ColorMethod, rhs: ColorMethod) -> Bool {
            switch (lhs, rhs) {
            case (.wCoordinate, .wCoordinate), (.distance, .distance), (.normal, .normal),
                (.curvature, .curvature), (.cell, .cell):
                return true
            case (.custom, .custom):
                // Functions can't be compared for equality, so we consider them equal
                // if they're both custom (this is a simplification)
                return true
            default:
                return false
            }
        }

        // Implement Hashable for custom case with function
        public func hash(into hasher: inout Hasher) {
            switch self {
            case .wCoordinate: hasher.combine(0)
            case .distance: hasher.combine(1)
            case .normal: hasher.combine(2)
            case .curvature: hasher.combine(3)
            case .cell: hasher.combine(4)
            case .custom: hasher.combine(5)
            }
        }
    }

    // MARK: - Properties

    /// The current color method
    private var colorMethod: ColorMethod = .wCoordinate

    /// Color map for w-coordinate visualization
    private var wColorMap: [(position: Float, color: SIMD4<Float>)] = [
        (-1.0, SIMD4<Float>(0.0, 0.0, 0.5, 1.0)),  // Deep blue
        (-0.5, SIMD4<Float>(0.0, 0.5, 1.0, 1.0)),  // Blue
        (0.0, SIMD4<Float>(1.0, 1.0, 1.0, 1.0)),  // White
        (0.5, SIMD4<Float>(1.0, 0.5, 0.0, 1.0)),  // Orange
        (1.0, SIMD4<Float>(0.5, 0.0, 0.0, 1.0)),  // Deep red
    ]

    /// The 4D vertices to visualize
    private var vertices4D: [SIMD4<Float>] = []

    /// The edges connecting vertices
    private var edges: [(Int, Int)] = []

    /// The triangular faces
    private var faces: [[Int]] = []

    /// The cells (volumetric elements)
    private var cells: [[[Int]]] = []

    /// Flag to use smooth normals
    private var useSmoothNormals: Bool = true

    /// Flag to highlight edges
    private var highlightEdges: Bool = true

    /// Edge highlighting color
    private var edgeColor: SIMD4<Float> = SIMD4<Float>(0.0, 0.0, 0.0, 1.0)

    /// Edge highlighting thickness
    private var edgeThickness: Float = 0.01

    // MARK: - Initialization

    /// Initialize the visualizer with 4D vertices
    /// - Parameters:
    ///   - vertices: The 4D vertices
    ///   - edges: The edges connecting vertices
    ///   - faces: Optional triangular faces
    ///   - cells: Optional 3D cells
    public init(
        vertices: [SIMD4<Float>], edges: [(Int, Int)], faces: [[Int]] = [], cells: [[[Int]]] = []
    ) {
        self.vertices4D = vertices
        self.edges = edges
        self.faces = faces
        self.cells = cells
    }

    // MARK: - Configuration

    /// Set the coloring method
    /// - Parameter method: The coloring method to use
    public func setColorMethod(_ method: ColorMethod) {
        self.colorMethod = method
    }

    /// Set the color map for w-coordinate coloring
    /// - Parameter colorMap: Array of (position, color) pairs
    public func setWColorMap(_ colorMap: [(position: Float, color: SIMD4<Float>)]) {
        self.wColorMap = colorMap
    }

    /// Set whether to use smooth normals
    /// - Parameter smooth: True to use smooth normals, false for flat normals
    public func setUseSmoothNormals(_ smooth: Bool) {
        self.useSmoothNormals = smooth
    }

    /// Configure edge highlighting
    /// - Parameters:
    ///   - highlight: Whether to highlight edges
    ///   - color: The color to use for edges
    ///   - thickness: The thickness of highlighted edges
    public func configureEdgeHighlighting(
        highlight: Bool, color: SIMD4<Float>? = nil, thickness: Float? = nil
    ) {
        self.highlightEdges = highlight

        if let color = color {
            self.edgeColor = color
        }

        if let thickness = thickness {
            self.edgeThickness = thickness
        }
    }

    /// Update the 4D vertices
    /// - Parameter vertices: The new 4D vertices
    public func updateVertices(_ vertices: [SIMD4<Float>]) {
        precondition(vertices.count == vertices4D.count, "Must provide same number of vertices")
        self.vertices4D = vertices
    }

    // MARK: - Visualization Methods

    /// Generate visualization data for the current 4D object
    /// - Parameters:
    ///   - rotors: Rotors to apply to the 4D object
    ///   - projectionType: The type of 4D to 3D projection to use
    /// - Returns: VisualizationData for rendering
    public func generateVisualization(
        rotors: [GA4D.Metric.Multivector],
        projectionType: GA4D.GA4DMetalBridge.ProjectionType = .stereographic
    ) -> Graf.VisualizationData {
        // Transform vertices using rotors
        let transformedVertices4D = transformVertices(vertices4D, rotors: rotors)

        // Project to 3D
        let vertices3D = project4Dto3D(transformedVertices4D, projectionType: projectionType)

        // Generate colors based on the selected method
        let colors = generateColors(for: transformedVertices4D)

        // Generate normals (either smooth or flat)
        let normals = generateNormals(vertices3D: vertices3D, vertices4D: transformedVertices4D)

        // Return the visualization data
        return Graf.VisualizationData(
            vertices: vertices3D,
            normals: normals,
            colors: colors,
            indices: [],  // Add this missing parameter
            edges: edges,
            faces: faces,
            originalVertices4D: transformedVertices4D
        )
    }

    /// Generate Metal-compatible render vertices
    /// - Parameter visualization: The visualization data
    /// - Returns: Array of RenderVertex structures for Metal rendering
    public func generateRenderVertices(from visualization: Graf.VisualizationData) -> [RenderVertex] {
        var renderVertices: [RenderVertex] = []

        // Create render vertices for each 3D vertex
        for i in 0..<visualization.vertices3D.count {
            let renderVertex = RenderVertex(
                position: visualization.vertices3D[i],
                normal: visualization.normals[i],
                color: visualization.colors[i],
                w: visualization.originalVertices4D[i].w
            )

            renderVertices.append(renderVertex)
        }

        return renderVertices
    }

    /// Generate render vertices for edges (for separate edge rendering)
    /// - Parameter visualization: The visualization data
    /// - Returns: Array of RenderVertex structures for edge rendering
    public func generateEdgeRenderVertices(from visualization: Graf.VisualizationData) -> [RenderVertex]
    {
        var edgeVertices: [RenderVertex] = []

        guard highlightEdges else { return [] }

        // Create vertices for each edge
        for (i, j) in visualization.edges {
            // Get the vertices at each end of the edge
            let v1 = visualization.vertices3D[i]
            let v2 = visualization.vertices3D[j]

            // Get the corresponding 4D vertices
            let v1_4D = visualization.originalVertices4D[i]
            let v2_4D = visualization.originalVertices4D[j]

            // Create two vertices for each edge
            let edgeVertex1 = RenderVertex(
                position: v1,
                normal: normalize(v2 - v1),  // Use edge direction as normal
                color: edgeColor,
                w: v1_4D.w
            )

            let edgeVertex2 = RenderVertex(
                position: v2,
                normal: normalize(v1 - v2),  // Use edge direction as normal
                color: edgeColor,
                w: v2_4D.w
            )

            edgeVertices.append(edgeVertex1)
            edgeVertices.append(edgeVertex2)
        }

        return edgeVertices
    }

    // MARK: - Helper Methods

    /// Transform 4D vertices using GA rotors
    /// - Parameters:
    ///   - vertices: Vertices to transform
    ///   - rotors: Rotors to apply
    /// - Returns: Transformed 4D vertices
    private func transformVertices(_ vertices: [SIMD4<Float>], rotors: [GA4D.Metric.Multivector])
        -> [SIMD4<Float>]
    {
        guard !rotors.isEmpty else { return vertices }

        return vertices.map { vertex in
            GA4D.Metric.Operations4D.applyRotorSequence(to: vertex, rotors: rotors)
        }
    }

    /// Project 4D vertices to 3D using the specified projection type
    /// - Parameters:
    ///   - vertices4D: 4D vertices to project
    ///   - projectionType: The type of projection to use
    /// - Returns: Projected 3D vertices
    private func project4Dto3D(
        _ vertices4D: [SIMD4<Float>], projectionType: GA4D.GA4DMetalBridge.ProjectionType
    ) -> [SIMD3<Float>] {
        return vertices4D.map { vertex4D in
            switch projectionType {
            case .stereographic:
                // Stereographic projection
                let factor = 1.0 / (1.0 - vertex4D.w * 0.1)
                return SIMD3<Float>(vertex4D.x, vertex4D.y, vertex4D.z) * factor

            case .perspective:
                // Perspective projection
                let distance: Float = 5.0
                let factor = distance / (distance + vertex4D.w)
                return SIMD3<Float>(vertex4D.x, vertex4D.y, vertex4D.z) * factor

            case .orthographic:
                // Orthographic projection (simply drop the w coordinate)
                return SIMD3<Float>(vertex4D.x, vertex4D.y, vertex4D.z)
            }
        }
    }

    /// Generate colors for 4D vertices based on the selected color method
    /// - Parameter vertices4D: The 4D vertices to color
    /// - Returns: Array of RGBA colors
    private func generateColors(for vertices4D: [SIMD4<Float>]) -> [SIMD4<Float>] {
        return vertices4D.map { vertex4D in
            switch colorMethod {
            case .wCoordinate:
                return colorFromWCoordinate(vertex4D.w)

            case .distance:
                let distance = simd_length(vertex4D)
                return colorFromDistance(distance)

            case .normal:
                let normal = normalize(vertex4D)
                return colorFromNormal(normal)

            case .curvature:
                // Curvature calculation would require more context
                // Return a default color for now
                return SIMD4<Float>(0.7, 0.7, 0.7, 1.0)

            case .cell:
                // Cell-based coloring would require cell info
                // Return a default color for now
                return SIMD4<Float>(0.7, 0.7, 0.7, 1.0)

            case .custom(let colorFunction):
                return colorFunction(vertex4D)
            }
        }
    }

    /// Generate vertex normals for the 3D projection
    /// - Parameters:
    ///   - vertices3D: Projected 3D vertices
    ///   - vertices4D: Original 4D vertices
    /// - Returns: Array of 3D normal vectors
    private func generateNormals(vertices3D: [SIMD3<Float>], vertices4D: [SIMD4<Float>]) -> [SIMD3<
        Float
    >] {
        if useSmoothNormals {
            return generateSmoothNormals(vertices3D: vertices3D, vertices4D: vertices4D)
        } else {
            return generateFlatNormals(vertices3D: vertices3D)
        }
    }

    /// Generate smooth normals by averaging face normals
    private func generateSmoothNormals(vertices3D: [SIMD3<Float>], vertices4D: [SIMD4<Float>])
        -> [SIMD3<Float>]
    {
        // Start with zero normals for each vertex
        var normals = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: vertices3D.count)

        // If we have face information, use it to compute normals
        if !faces.isEmpty {
            // For each face, compute normal and add to vertex normals
            for face in faces {
                if face.count >= 3 {
                    // Get three vertices to compute a normal
                    let v1 = vertices3D[face[0]]
                    let v2 = vertices3D[face[1]]
                    let v3 = vertices3D[face[2]]

                    // Compute face normal using cross product
                    let edge1 = v2 - v1
                    let edge2 = v3 - v1
                    let normal = normalize(cross(edge1, edge2))

                    // Add to each vertex normal
                    for vertexIndex in face {
                        normals[vertexIndex] += normal
                    }
                }
            }
        } else {
            // If no face information, use 4D vertex direction as normal
            for i in 0..<vertices4D.count {
                let normal4D = normalize(vertices4D[i])
                normals[i] = SIMD3<Float>(normal4D.x, normal4D.y, normal4D.z)
            }
        }

        // Normalize all normals
        for i in 0..<normals.count {
            let length = simd_length(normals[i])
            if length > 1e-6 {
                normals[i] /= length
            } else {
                // Default normal if no faces use this vertex
                normals[i] = SIMD3<Float>(0, 0, 1)
            }
        }

        return normals
    }

    /// Generate flat normals (per face)
    private func generateFlatNormals(vertices3D: [SIMD3<Float>]) -> [SIMD3<Float>] {
        // If no face information, use vertex direction as normal
        if faces.isEmpty {
            return vertices3D.map { normalize($0) }
        }

        // Start with default normals
        var normals = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 1), count: vertices3D.count)

        // For each face, compute normal
        for face in faces {
            if face.count >= 3 {
                // Get three vertices to compute a normal
                let v1 = vertices3D[face[0]]
                let v2 = vertices3D[face[1]]
                let v3 = vertices3D[face[2]]

                // Compute face normal using cross product
                let edge1 = v2 - v1
                let edge2 = v3 - v1
                let normal = normalize(cross(edge1, edge2))

                // Set normal for each vertex in the face
                for vertexIndex in face {
                    normals[vertexIndex] = normal
                }
            }
        }

        return normals
    }

    /// Generate a color from w-coordinate using the color map
    private func colorFromWCoordinate(_ w: Float) -> SIMD4<Float> {
        // Clamp w to the range [-1, 1]
        let clampedW = max(-1.0, min(1.0, w))

        // Find the two color points around our w value
        var leftIndex = 0
        var rightIndex = 1

        while rightIndex < wColorMap.count && wColorMap[rightIndex].position < clampedW {
            leftIndex = rightIndex
            rightIndex += 1
        }

        // If w is past the last point, use the last color
        if rightIndex >= wColorMap.count {
            return wColorMap[leftIndex].color
        }

        // Interpolate between the two colors
        let leftPos = wColorMap[leftIndex].position
        let rightPos = wColorMap[rightIndex].position
        let leftColor = wColorMap[leftIndex].color
        let rightColor = wColorMap[rightIndex].color

        // Calculate interpolation factor
        let t = (clampedW - leftPos) / (rightPos - leftPos)

        // Interpolate color components
        return mix(leftColor, rightColor, t: t)
    }

    /// Generate a color from distance from origin
    private func colorFromDistance(_ distance: Float) -> SIMD4<Float> {
        // Map distance to a color value (using a simple gradient)
        let normalizedDistance = min(1.0, distance / 2.0)  // Assuming typical distances are 0-2

        // Create a color from blue to red
        return SIMD4<Float>(normalizedDistance, 0.2, 1.0 - normalizedDistance, 1.0)
    }

    /// Generate a color from a normal vector
    private func colorFromNormal(_ normal: SIMD4<Float>) -> SIMD4<Float> {
        // Map normal components to RGB (mapping from [-1,1] to [0,1])
        let r = (normal.x + 1.0) * 0.5
        let g = (normal.y + 1.0) * 0.5
        let b = (normal.z + 1.0) * 0.5

        return SIMD4<Float>(r, g, b, 1.0)
    }
}

// MARK: - SIMD Utility Extensions

extension SIMD4<Float> {
    /// Linear interpolation between two SIMD4<Float> values
    static func lerp(_ a: SIMD4<Float>, _ b: SIMD4<Float>, t: Float) -> SIMD4<Float> {
        return a + t * (b - a)
    }
}

/// Linear interpolation for SIMD4<Float>
func mix(_ a: SIMD4<Float>, _ b: SIMD4<Float>, t: Float) -> SIMD4<Float> {
    return SIMD4<Float>.lerp(a, b, t: t)
}

///// Visualization data for rendering
//public struct VisualizationData {
//    /// 3D projected vertices
//    public let vertices3D: [SIMD3<Float>]
//
//    /// Vertex colors (RGBA)
//    public let colors: [SIMD4<Float>]
//
//    /// Vertex normals
//    public let normals: [SIMD3<Float>]
//
//    /// Edge connections
//    public let edges: [(Int, Int)]
//
//    /// Triangular faces
//    public let faces: [[Int]]
//
//    /// Original 4D data (for reference)
//    public let originalVertices4D: [SIMD4<Float>]
//}

/// Vertex with all attributes for Metal rendering
public struct RenderVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var color: SIMD4<Float>
    var texCoord: SIMD2<Float>
    var w: Float  // Original w coordinate for visualization

    init(position: SIMD3<Float>, normal: SIMD3<Float>, color: SIMD4<Float>, w: Float) {
        self.position = position
        self.normal = normal
        self.color = color
        self.texCoord = SIMD2<Float>(0, 0)  // Default texture coordinates
        self.w = w
    }
}
