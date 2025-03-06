//
//  GA4DPrimitives.swift
//  Graf
//

import Foundation
import simd

/// Extension to GA4D to provide advanced 4D primitive shapes
extension GA4D {
    /// Advanced 4D primitives beyond the basic tesseract and hypersphere
    public struct AdvancedPrimitives {
        // MARK: - 24-Cell

        /// Creates vertices for a 24-cell, a self-dual 4D polytope with 24 octahedral cells
        /// - Parameter scale: Size scaling factor
        /// - Returns: Array of 4D vertices
        public static func create24Cell(scale: Float = 1.0) -> [SIMD4<Float>] {
            var vertices: [SIMD4<Float>] = []

            // The 24-cell has 24 vertices, which can be described as:
            // 1. The 8 vertices with coordinates (±1, 0, 0, 0) and permutations
            // 2. The 16 vertices with coordinates (±0.5, ±0.5, ±0.5, ±0.5)

            // Add the 8 vertices of type (±1, 0, 0, 0) and permutations
            for i in 0..<4 {
                for s in [-1.0, 1.0] {
                    var vertex = SIMD4<Float>(0, 0, 0, 0)
                    vertex[i] = Float(s)
                    vertices.append(vertex * scale)
                }
            }

            // Add the 16 vertices of type (±0.5, ±0.5, ±0.5, ±0.5)
            for x in [-0.5, 0.5] {
                for y in [-0.5, 0.5] {
                    for z in [-0.5, 0.5] {
                        for w in [-0.5, 0.5] {
                            // Only include vertices where the product of all signs is positive
                            // (This ensures we get exactly 16 vertices, not 2^4 = 32)
                            if x * y * z * w > 0 {
                                vertices.append(
                                    SIMD4<Float>(Float(x), Float(y), Float(z), Float(w)) * scale
                                        * 2.0)
                            }
                        }
                    }
                }
            }

            return vertices
        }

        /// Creates edges for a 24-cell
        /// - Parameter vertices: Array of 24-cell vertices
        /// - Returns: Array of index pairs defining edges
        public static func create24CellEdges(vertices: [SIMD4<Float>]) -> [(Int, Int)] {
            var edges: [(Int, Int)] = []

            // Connect vertices if they are at distance 1 from each other
            for i in 0..<vertices.count {
                for j in (i + 1)..<vertices.count {
                    let distance = simd_distance(vertices[i], vertices[j])
                    // Use a threshold to account for floating-point precision
                    if abs(distance - 1.0) < 0.01 {
                        edges.append((i, j))
                    }
                }
            }

            return edges
        }

        // MARK: - 120-Cell

        /// Creates vertices for a 120-cell, a regular 4D polytope with 120 dodecahedral cells
        /// Note: This is a simplified version with fewer vertices for performance
        /// - Parameter scale: Size scaling factor
        /// - Returns: Array of 4D vertices
        public static func create120Cell(scale: Float = 1.0) -> [SIMD4<Float>] {
            var vertices: [SIMD4<Float>] = []

            // The full 120-cell has 600 vertices, but for visualization we'll use a simplified version
            // based on the snub 24-cell construction (which gives a chiral approximation)

            // Start with the 24-cell vertices
            let baseCellVertices = create24Cell(scale: scale * 0.5)
            vertices.append(contentsOf: baseCellVertices)

            // Create rotated copies of the 24-cell
            let rotationAngles: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]

            for angle in rotationAngles {
                // Create rotors for different 4D rotations
                let rotorXY = GA4D.Metric.Operations4D.rotationXY(angle: angle)
                let rotorZW = GA4D.Metric.Operations4D.rotationZW(angle: angle * 1.618)  // Golden ratio for interesting patterns

                // Combine rotors
                for vertex in baseCellVertices {
                    // Apply rotors
                    let rotated = GA4D.Metric.Operations4D.rotate(
                        vector: GA4D.Metric.Operations4D.rotate(vector: vertex, using: rotorXY),
                        using: rotorZW
                    )

                    // Add rotated vertex (avoid duplicates)
                    var isDuplicate = false
                    for existingVertex in vertices {
                        if simd_distance(rotated, existingVertex) < 0.01 {
                            isDuplicate = true
                            break
                        }
                    }

                    if !isDuplicate {
                        vertices.append(rotated)
                    }
                }
            }

            return vertices
        }

        // MARK: - 4D Torus

        /// Creates vertices for a 4D torus (different from a Clifford torus)
        /// This models a traditional 3D torus rotated through 4D
        /// - Parameters:
        ///   - majorRadius: The major radius of the torus
        ///   - minorRadius: The minor radius of the tube
        ///   - resolution: Number of divisions in each circular dimension
        /// - Returns: Array of 4D vertices
        public static func create4DTorus(
            majorRadius: Float = 1.0, minorRadius: Float = 0.3, resolution: Int = 24
        ) -> [SIMD4<Float>] {
            var vertices: [SIMD4<Float>] = []

            // Create a 3D torus parameterized by two angles
            let stepU = 2.0 * Float.pi / Float(resolution)
            let stepV = 2.0 * Float.pi / Float(resolution)

            for i in 0..<resolution {
                let u = Float(i) * stepU

                for j in 0..<resolution {
                    let v = Float(j) * stepV

                    // Standard torus parameterization
                    let x = (majorRadius + minorRadius * cos(v)) * cos(u)
                    let y = (majorRadius + minorRadius * cos(v)) * sin(u)
                    let z = minorRadius * sin(v)

                    // Add a 4D rotation through the w axis
                    let angle = u * 0.5  // 4D rotation angle

                    // Create a 4D rotation matrix for XW plane
                    let rotorXW = GA4D.Metric.Operations4D.rotationXW(angle: angle)

                    // Rotate the 3D torus point through 4D
                    let baseVector = SIMD4<Float>(x, y, z, 0)
                    let rotated = GA4D.Metric.Operations4D.rotate(
                        vector: baseVector, using: rotorXW)

                    vertices.append(rotated)
                }
            }

            return vertices
        }

        // MARK: - 4D Hopf Fibration

        /// Creates a visualization of the Hopf fibration, which describes how S³ (3-sphere) is composed of circles (S¹)
        /// - Parameters:
        ///   - radius: Overall radius of the 3-sphere
        ///   - fiberCount: Number of fibers to generate
        ///   - pointsPerFiber: Number of points per fiber
        /// - Returns: Array of 4D vertices
        public static func createHopfFibration(
            radius: Float = 1.0, fiberCount: Int = 12, pointsPerFiber: Int = 24
        ) -> [SIMD4<Float>] {
            var vertices: [SIMD4<Float>] = []

            // Generate several fibers (circles) in S³
            for f in 0..<fiberCount {
                // Parameter for this fiber
                let phi = Float.pi * Float(f) / Float(fiberCount)

                // Base point on S² (will generate a fiber from this)
                let basePoint = SIMD3<Float>(
                    sin(phi) * cos(Float(f) * 0.5),
                    sin(phi) * sin(Float(f) * 0.5),
                    cos(phi)
                )

                // Generate a fiber (circle) above this base point
                for p in 0..<pointsPerFiber {
                    let t = 2.0 * Float.pi * Float(p) / Float(pointsPerFiber)

                    // Use quaternion representation for points on S³
                    // A point on S³ can be written as (cos(t/2), sin(t/2)*v) where v is a unit vector in R³
                    let x = cos(t / 2)
                    let y = sin(t / 2) * basePoint.x
                    let z = sin(t / 2) * basePoint.y
                    let w = sin(t / 2) * basePoint.z

                    vertices.append(SIMD4<Float>(x, y, z, w) * radius)
                }
            }

            return vertices
        }

        // MARK: - 4D Julia Set Fractal

        /// Creates a 4D slice of a quaternionic Julia set
        /// - Parameters:
        ///   - resolution: Grid resolution for the sampling
        ///   - c: The quaternion parameter for the Julia set
        ///   - maxIterations: Maximum iterations for convergence testing
        /// - Returns: Array of 4D vertices representing points in the Julia set
        public static func createJuliaSet(
            resolution: Int = 20, c: SIMD4<Float> = SIMD4<Float>(0.0, 0.7, 0.3, 0.1),
            maxIterations: Int = 10
        ) -> [SIMD4<Float>] {
            var vertices: [SIMD4<Float>] = []

            // Create a 4D grid of sample points
            let stepSize: Float = 2.0 / Float(resolution)
            let range = stride(from: Float(-1.0), through: Float(1.0), by: Float(stepSize))

            for x in range {
                for y in range {
                    for z in range {
                        // For 4D, we'll only use a slice along w=0 for performance
                        let q = SIMD4<Float>(Float(x), Float(y), Float(z), 0.0)

                        // Check if this point is in the Julia set
                        if isInJuliaSet(q: q, c: c, maxIterations: maxIterations) {
                            vertices.append(q)
                        }
                    }
                }
            }

            return vertices
        }

        // Helper function to check if a quaternion point is in the Julia set
        private static func isInJuliaSet(q: SIMD4<Float>, c: SIMD4<Float>, maxIterations: Int)
            -> Bool
        {
            var z = q

            for _ in 0..<maxIterations {
                // Calculate quaternion square
                z = quaternionSquare(z) + c

                // Check if the point escapes
                if simd_length_squared(z) > 4.0 {
                    return false
                }
            }

            return true
        }

        // Helper function for quaternion multiplication
        private static func quaternionMultiply(_ q1: SIMD4<Float>, _ q2: SIMD4<Float>) -> SIMD4<
            Float
        > {
            let a1 = q1.x
            let b1 = q1.y
            let c1 = q1.z
            let d1 = q1.w

            let a2 = q2.x
            let b2 = q2.y
            let c2 = q2.z
            let d2 = q2.w

            return SIMD4<Float>(
                a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
            )
        }

        // Helper function to square a quaternion
        private static func quaternionSquare(_ q: SIMD4<Float>) -> SIMD4<Float> {
            return quaternionMultiply(q, q)
        }
    }
}
