//
//  GA4DIntegration.swift
//  Graf
//
//  Created by HAWZHIN on 05/03/2025.
//

import Foundation
import simd

/// Integration utilities to bridge between GA4D and the Graf application
class GA4DIntegration {

    /// Convert between GA4D and Graf projection types
    class func convertProjectionType(_ type: GA4D.GA4DMetalBridge.ProjectionType)
        -> Graf.ProjectionType
    {
        switch type {
        case .stereographic:
            return .stereographic
        case .perspective:
            return .perspective
        case .orthographic:
            return .orthographic
        }
    }

    /// Convert from Graf to GA4D projection types
    class func convertGrafProjectionType(_ type: Graf.ProjectionType)
        -> GA4D.GA4DMetalBridge.ProjectionType
    {
        switch type {
        case .stereographic:
            return .stereographic
        case .perspective:
            return .perspective
        case .orthographic:
            return .orthographic
        }
    }

    /// Generate a Graf.VisualizationData from GA4D primitives
    class func createVisualizationData(
        type: Graf.VisualizationType,
        resolution: Int,
        scale: Float,
        rotations: (xy: Float, xz: Float, xw: Float, yz: Float, yw: Float, zw: Float)
    ) -> Graf.VisualizationData {

        // Create 4D vertices based on the visualization type
        var vertices4D: [SIMD4<Float>] = []
        var edges: [(Int, Int)] = []
        var faces: [[Int]] = []

        switch type {
        case .tesseract:
            vertices4D = GA4D.AdvancedPrimitives.create24Cell(scale: scale)
            edges = GA4D.AdvancedPrimitives.create24CellEdges(vertices: vertices4D)

            // Create faces for the tesseract (simplified approach)
            for i in 0..<vertices4D.count {
                for j in (i + 1)..<vertices4D.count {
                    for k in (j + 1)..<vertices4D.count {
                        // Check if i, j, k form a potential face corner
                        if edges.contains(where: { ($0 == i && $1 == j) || ($0 == j && $1 == i) })
                            && edges.contains(where: {
                                ($0 == j && $1 == k) || ($0 == k && $1 == j)
                            })
                        {

                            // Look for a fourth vertex to complete the face
                            for l in (k + 1)..<vertices4D.count {
                                if edges.contains(where: {
                                    ($0 == k && $1 == l) || ($0 == l && $1 == k)
                                })
                                    && edges.contains(where: {
                                        ($0 == l && $1 == i) || ($0 == i && $1 == l)
                                    })
                                {

                                    // Verify it's a valid face by checking coplanarity in 4D
                                    // This is a simplified check
                                    let v1 = vertices4D[i]
                                    let v2 = vertices4D[j]
                                    let v3 = vertices4D[k]
                                    let v4 = vertices4D[l]

                                    // Count dimensions with same value across all vertices
                                    var matchingDimensions = 0

                                    // Check X coordinates
                                    if abs(v1.x - v2.x) < 0.01 && abs(v1.x - v3.x) < 0.01
                                        && abs(v1.x - v4.x) < 0.01
                                    {
                                        matchingDimensions += 1
                                    }

                                    // Check Y coordinates
                                    if abs(v1.y - v2.y) < 0.01 && abs(v1.y - v3.y) < 0.01
                                        && abs(v1.y - v4.y) < 0.01
                                    {
                                        matchingDimensions += 1
                                    }

                                    // Check Z coordinates
                                    if abs(v1.z - v2.z) < 0.01 && abs(v1.z - v3.z) < 0.01
                                        && abs(v1.z - v4.z) < 0.01
                                    {
                                        matchingDimensions += 1
                                    }

                                    // Check W coordinates
                                    if abs(v1.w - v2.w) < 0.01 && abs(v1.w - v3.w) < 0.01
                                        && abs(v1.w - v4.w) < 0.01
                                    {
                                        matchingDimensions += 1
                                    }

                                    // A face in 4D has 2 fixed dimensions
                                    if matchingDimensions == 2 {
                                        faces.append([i, j, k, l])
                                    }
                                }
                            }
                        }
                    }
                }
            }

        case .hypersphere:
            vertices4D = GA4D.AdvancedPrimitives.createHopfFibration(
                radius: scale, fiberCount: resolution, pointsPerFiber: resolution)

            // Generate edges for the hypersphere
            let connectionThreshold = scale * 0.3
            for i in 0..<vertices4D.count {
                for j in (i + 1)..<min(i + 30, vertices4D.count) {
                    if simd_distance(vertices4D[i], vertices4D[j]) < connectionThreshold {
                        edges.append((i, j))
                    }
                }
            }

            // Generate triangular faces for the hypersphere
            for (i, j) in edges {
                for k in 0..<vertices4D.count {
                    if k != i && k != j {
                        if edges.contains(where: { ($0 == i && $1 == k) || ($0 == k && $1 == i) })
                            && edges.contains(where: {
                                ($0 == j && $1 == k) || ($0 == k && $1 == j)
                            })
                        {
                            faces.append([i, j, k])
                        }
                    }
                }
            }

        case .cliffordTorus:
            vertices4D = GA4D.AdvancedPrimitives.create4DTorus(
                majorRadius: scale,
                minorRadius: scale * 0.5,
                resolution: resolution
            )

            // Generate grid-like edges for the Clifford torus
            for i in 0..<resolution {
                for j in 0..<resolution {
                    let index = i * (resolution + 1) + j

                    // Connect to next point in u direction
                    if j < resolution {
                        edges.append((index, index + 1))
                    }

                    // Connect to next point in v direction
                    if i < resolution {
                        edges.append((index, index + (resolution + 1)))
                    }

                    // Create quad faces
                    if i < resolution - 1 && j < resolution - 1 {
                        faces.append([
                            index,
                            index + 1,
                            index + resolution + 2,
                            index + resolution + 1,
                        ])
                    }
                }
            }

        case .duocylinder:
            // Create a duocylinder (product of two disks)
            let segments = max(4, resolution / 4)
            let radius = scale * 0.5

            // Generate vertices
            for a in 0...segments {
                let angleA = Float(a) / Float(segments) * (2 * Float.pi)

                for r1 in 0...segments / 2 {
                    let radius1 = Float(r1) / Float(segments / 2) * radius

                    for b in 0...segments {
                        let angleB = Float(b) / Float(segments) * (2 * Float.pi)

                        for r2 in 0...segments / 2 {
                            let radius2 = Float(r2) / Float(segments / 2) * radius

                            // Skip some points for better performance
                            if r1 * r2 > 0 && (r1 + r2) % 2 != 0 && a % 2 == 0 && b % 2 == 0 {
                                continue
                            }

                            // Calculate position using disk parametrization
                            let x = radius1 * cos(angleA)
                            let y = radius1 * sin(angleA)
                            let z = radius2 * cos(angleB)
                            let w = radius2 * sin(angleB)

                            vertices4D.append(SIMD4<Float>(x, y, z, w))
                        }
                    }
                }
            }

            // Generate edges (nearest neighbors)
            for i in 0..<vertices4D.count {
                for j in (i + 1)..<min(i + 20, vertices4D.count) {
                    if simd_distance(vertices4D[i], vertices4D[j]) < radius * 0.3 {
                        edges.append((i, j))
                    }
                }
            }

        case .quaternion:
            // Generate quaternion visualization
            let segments = max(5, resolution / 6)

            // Sample unit quaternions on a 4D grid
            for w in -segments...segments {
                let wVal = Float(w) / Float(segments)

                for x in -segments...segments {
                    let xVal = Float(x) / Float(segments)

                    for y in -segments...segments {
                        let yVal = Float(y) / Float(segments)

                        // Calculate z to ensure unit quaternion (|q| = 1)
                        let sqSum = wVal * wVal + xVal * xVal + yVal * yVal
                        if sqSum <= 1.0 {
                            let zVal = sqrt(1.0 - sqSum)

                            // Add both positive and negative z solutions
                            vertices4D.append(SIMD4<Float>(wVal, xVal, yVal, zVal) * scale)

                            if zVal != 0 {
                                vertices4D.append(SIMD4<Float>(wVal, xVal, yVal, -zVal) * scale)
                            }
                        }
                    }
                }
            }

            // Generate edges
            for i in 0..<vertices4D.count {
                for j in (i + 1)..<min(i + 10, vertices4D.count) {
                    if simd_distance(vertices4D[i], vertices4D[j]) < 0.2 * scale {
                        edges.append((i, j))
                    }
                }
            }

        case .customFunction:
            // For custom function, we'll create a basic shape that can be modified later
            let gridSize = max(8, resolution / 4)
            let step = 2.0 * scale / Float(gridSize)

            // Create a grid in the XY plane
            for i in 0...gridSize {
                for j in 0...gridSize {
                    let x = -scale + Float(i) * step
                    let y = -scale + Float(j) * step

                    // Use a sine function for z and w coordinates
                    let z = sin(x * 3) * cos(y * 2) * 0.3 * scale
                    let w = cos(x * 2) * sin(y * 3) * 0.3 * scale

                    vertices4D.append(SIMD4<Float>(x, y, z, w))
                }
            }

            // Generate grid edges
            let width = gridSize + 1
            for i in 0..<width {
                for j in 0..<width {
                    let index = i * width + j

                    // Connect to next point in x direction
                    if j < width - 1 {
                        edges.append((index, index + 1))
                    }

                    // Connect to next point in y direction
                    if i < width - 1 {
                        edges.append((index, index + width))
                    }

                    // Create quad faces
                    if i < width - 1 && j < width - 1 {
                        faces.append([
                            index,
                            index + 1,
                            index + width + 1,
                            index + width,
                        ])
                    }
                }
            }

        default:
            // Default to a simple 4D cube
            vertices4D = GA4D.AdvancedPrimitives.create24Cell(scale: scale)
            edges = GA4D.AdvancedPrimitives.create24CellEdges(vertices: vertices4D)
        }

        // Apply 4D rotations using GA4D
        let projType = GA4D.GA4DMetalBridge.ProjectionType.stereographic

        // Transform to 3D
        let vertices3D = GA4D.GA4DMetalBridge.transformVertices(
            vertices4D: vertices4D,
            rotations: rotations,
            projectionType: projType
        )

        // Generate normals
        var triangleFaces: [(Int, Int, Int)] = []

        // Convert quad faces to triangles for normal calculation
        for face in faces {
            if face.count == 3 {
                triangleFaces.append((face[0], face[1], face[2]))
            } else if face.count >= 4 {
                // Triangulate quad or n-gon
                for i in 1..<(face.count - 1) {
                    triangleFaces.append((face[0], face[i], face[i + 1]))
                }
            }
        }

        let normals4D = GA4D.GA4DMetalBridge.calculateNormals(
            for: vertices4D, connections: triangleFaces)

        // Project normals to 3D
        let normals3D = normals4D.map { normal4D in
            SIMD3<Float>(normal4D.x, normal4D.y, normal4D.z)
        }

        // Generate colors based on 4D position
        let colors = vertices4D.map { v in
            // Create a color based on 4D position
            let maxCoord = max(abs(v.x), max(abs(v.y), max(abs(v.z), abs(v.w))))
            let normalizedX = (v.x / maxCoord + 1) * 0.5
            let normalizedY = (v.y / maxCoord + 1) * 0.5
            let normalizedZ = (v.z / maxCoord + 1) * 0.5
            let normalizedW = (v.w / maxCoord + 1) * 0.5

            // Blend colors based on all 4 coordinates
            return SIMD4<Float>(
                normalizedX * 0.8 + normalizedW * 0.2,
                normalizedY * 0.8 + normalizedW * 0.2,
                normalizedZ * 0.8 + normalizedW * 0.2,
                1.0
            )
        }

        // Create indices for rendering
        var indices: [UInt32] = []
        for (start, end) in edges {
            indices.append(UInt32(start))
            indices.append(UInt32(end))
        }

        // Return complete visualization data
        return Graf.VisualizationData(
            vertices: vertices3D,
            normals: normals3D,
            colors: colors,
            indices: indices,
            edges: edges,
            faces: faces,
            originalVertices4D: vertices4D
        )
    }

    /// Generate a Graf.VisualizationData for a custom function
    class func createCustomFunctionVisualization(
        expression: String,
        resolution: Int,
        scale: Float,
        rotations: (xy: Float, xz: Float, xw: Float, yz: Float, yw: Float, zw: Float)
    ) -> Graf.VisualizationData {
        // Create a grid of points
        let gridSize = max(8, resolution / 4)
        let step = 2.0 * scale / Float(gridSize)

        // Variables to store generated data
        var vertices4D: [SIMD4<Float>] = []
        var edges: [(Int, Int)] = []
        var faces: [[Int]] = []

        // Create expressions evaluator
        let evaluator = ExpressionEvaluator()

        // Generate grid points and evaluate function
        for i in 0...gridSize {
            for j in 0...gridSize {
                let x = -scale + Float(i) * step
                let y = -scale + Float(j) * step

                // Evaluate custom function for z and w coordinates
                let z = evaluator.evaluate(expression: expression, x: x, y: y, z: 0, w: 0)
                let w = evaluator.evaluate(expression: expression, x: x, y: y, z: 0, w: 0.5)

                vertices4D.append(SIMD4<Float>(x, y, z, w))
            }
        }

        // Generate grid edges
        let width = gridSize + 1
        for i in 0..<width {
            for j in 0..<width {
                let index = i * width + j

                // Connect to next point in x direction
                if j < width - 1 {
                    edges.append((index, index + 1))
                }

                // Connect to next point in y direction
                if i < width - 1 {
                    edges.append((index, index + width))
                }

                // Create quad faces
                if i < width - 1 && j < width - 1 {
                    faces.append([
                        index,
                        index + 1,
                        index + width + 1,
                        index + width,
                    ])
                }
            }
        }

        // Apply 4D rotations and project to 3D
        let projType = GA4D.GA4DMetalBridge.ProjectionType.stereographic

        // Transform to 3D
        let vertices3D = GA4D.GA4DMetalBridge.transformVertices(
            vertices4D: vertices4D,
            rotations: rotations,
            projectionType: projType
        )

        // Generate triangular face connections for normal calculation
        var triangleFaces: [(Int, Int, Int)] = []

        // Convert quad faces to triangles
        for face in faces {
            if face.count >= 3 {
                // Triangulate as a fan from first vertex
                for i in 1..<(face.count - 1) {
                    triangleFaces.append((face[0], face[i], face[i + 1]))
                }
            }
        }

        // Calculate normals using GA4D
        let normals4D = GA4D.GA4DMetalBridge.calculateNormals(
            for: vertices4D, connections: triangleFaces)

        // Project normals to 3D
        let normals3D = normals4D.map { normal4D in
            SIMD3<Float>(normal4D.x, normal4D.y, normal4D.z)
        }

        // Generate colors based on function values
        let colors = vertices4D.map { v in
            // Color based on function value (z and w)
            let normalizedZ = (v.z / scale + 1) * 0.5
            let normalizedW = (v.w / scale + 1) * 0.5

            return SIMD4<Float>(
                normalizedZ,
                (normalizedZ + normalizedW) * 0.3,
                normalizedW,
                1.0
            )
        }

        // Create indices for rendering
        var indices: [UInt32] = []
        for (start, end) in edges {
            indices.append(UInt32(start))
            indices.append(UInt32(end))
        }

        // Return complete visualization data
        return Graf.VisualizationData(
            vertices: vertices3D,
            normals: normals3D,
            colors: colors,
            indices: indices,
            edges: edges,
            faces: faces,
            originalVertices4D: vertices4D
        )
    }
}

/// Helper class for evaluating mathematical expressions
class ExpressionEvaluator {
    // Simple expression evaluator for demo purposes
    // In a production app, use a proper expression parser library

    func evaluate(expression: String, x: Float, y: Float, z: Float, w: Float) -> Float {
        // Convert expression to lowercase for case-insensitive matching
        let expr = expression.lowercased()

        // Handle some common expressions
        if expr.contains("sin") && expr.contains("x") && expr.contains("cos") && expr.contains("y")
        {
            return sin(x) * cos(y)
        } else if expr.contains("sin") && expr.contains("sqrt") {
            return sin(sqrt(x * x + y * y + z * z + w * w))
        } else if expr.contains("x*x") || expr.contains("x^2") {
            return x * x - y * y + z * z - w * w
        } else if expr.contains("sin") && expr.contains("w") {
            return sin(x) * cos(y) * sin(w)
        } else if expr.contains("x*y") || expr.contains("z*w") {
            return sin(x * y + z * w)
        }

        // Default fallback for unknown expressions
        return sin(x * 3) * cos(y * 2)
    }
}
