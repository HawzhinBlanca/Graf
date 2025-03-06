import Foundation
import simd

/// Cross-sectioning system for 4D objects using Geometric Algebra
public class GA4DCrossSection {

    // MARK: - Types

    /// Type of hyperplane to use for cross-sectioning
    public enum HyperplaneType: Hashable, Equatable {
        case xConstant(Float)  // Hyperplane where x = constant
        case yConstant(Float)  // Hyperplane where y = constant
        case zConstant(Float)  // Hyperplane where z = constant
        case wConstant(Float)  // Hyperplane where w = constant
        case custom(SIMD4<Float>, Float)  // Custom hyperplane with normal and distance

        // Implement Equatable
        public static func == (lhs: HyperplaneType, rhs: HyperplaneType) -> Bool {
            switch (lhs, rhs) {
            case let (.xConstant(v1), .xConstant(v2)):
                return v1 == v2
            case let (.yConstant(v1), .yConstant(v2)):
                return v1 == v2
            case let (.zConstant(v1), .zConstant(v2)):
                return v1 == v2
            case let (.wConstant(v1), .wConstant(v2)):
                return v1 == v2
            case let (.custom(n1, d1), .custom(n2, d2)):
                return n1 == n2 && d1 == d2
            default:
                return false
            }
        }

        // Implement Hashable
        public func hash(into hasher: inout Hasher) {
            switch self {
            case .xConstant(let value):
                hasher.combine(0)  // Discriminator for xConstant
                hasher.combine(value)
            case .yConstant(let value):
                hasher.combine(1)  // Discriminator for yConstant
                hasher.combine(value)
            case .zConstant(let value):
                hasher.combine(2)  // Discriminator for zConstant
                hasher.combine(value)
            case .wConstant(let value):
                hasher.combine(3)  // Discriminator for wConstant
                hasher.combine(value)
            case .custom(let normal, let distance):
                hasher.combine(4)  // Discriminator for custom
                hasher.combine(normal.x)
                hasher.combine(normal.y)
                hasher.combine(normal.z)
                hasher.combine(normal.w)
                hasher.combine(distance)
            }
        }
    }

    /// Represents a 3D cross-section of a 4D object
    public struct CrossSection {
        /// 3D vertices representing the cross-section
        public let vertices: [SIMD3<Float>]

        /// Edges connecting the vertices (pairs of indices)
        public let edges: [(Int, Int)]

        /// Faces defined by vertex indices
        public let faces: [[Int]]

        /// Color information for visualization
        public let colors: [SIMD4<Float>]
    }

    // MARK: - Properties

    /// The vertices of the 4D object
    private var vertices4D: [SIMD4<Float>] = []

    /// The edges of the 4D object (pairs of vertex indices)
    private var edges4D: [(Int, Int)] = []

    /// The cells of the 4D object (each cell is a group of connected faces)
    private var cells4D: [[[Int]]] = []

    /// Current hyperplane for cross-sectioning
    private var hyperplane: HyperplaneType = .wConstant(0)

    /// Cross-section generation parameters
    private var intersectionThreshold: Float = 0.001
    private var maxSearchDistance: Float = 2.0

    // MARK: - Initialization

    /// Create a new cross-section system
    /// - Parameters:
    ///   - vertices: Vertices of the 4D object
    ///   - edges: Edges of the 4D object (pairs of vertex indices)
    ///   - cells: Optional cells of the 4D object
    public init(vertices: [SIMD4<Float>], edges: [(Int, Int)], cells: [[[Int]]]? = nil) {
        self.vertices4D = vertices
        self.edges4D = edges
        self.cells4D = cells ?? []
    }

    // MARK: - Public Methods

    /// Set the hyperplane to use for cross-sectioning
    /// - Parameter hyperplane: The hyperplane definition
    public func setHyperplane(_ hyperplane: HyperplaneType) {
        self.hyperplane = hyperplane
    }

    /// Update the vertices of the 4D object (e.g., after transformation)
    /// - Parameter vertices: New vertices
    public func updateVertices(_ vertices: [SIMD4<Float>]) {
        precondition(vertices.count == vertices4D.count, "Vertex count must match")
        self.vertices4D = vertices
    }

    /// Generate a 3D cross-section of the 4D object
    /// - Returns: The cross-section representation
    public func generateCrossSection() -> CrossSection {
        // Step 1: Extract hyperplane normal and distance
        let (normal, distance) = getHyperplaneParameters()

        // Step 2: Find edge intersections with the hyperplane
        let intersections = findEdgeIntersections(normal: normal, distance: distance)

        // Step 3: Determine edges of the cross-section
        let (sectionVertices, sectionEdges) = constructCrossSectionMesh(from: intersections)

        // Step 4: Generate colors based on position in 4D
        let colors = generateColors(for: sectionVertices)

        // Step 5: Create faces (simple triangulation)
        let faces = triangulateIntersection(sectionVertices, sectionEdges)

        return CrossSection(
            vertices: sectionVertices,
            edges: sectionEdges,
            faces: faces,
            colors: colors
        )
    }

    /// Animate the cross-section through the 4D object
    /// - Parameters:
    ///   - startValue: Starting position
    ///   - endValue: Ending position
    ///   - steps: Number of steps in the animation
    ///   - callback: Function called for each step with the cross-section
    public func animateSection(
        startValue: Float, endValue: Float, steps: Int,
        callback: @escaping (CrossSection) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            for step in 0..<steps {
                let t = Float(step) / Float(steps - 1)
                let value = startValue + t * (endValue - startValue)
                self.hyperplane = .xConstant(value)

                // Update the hyperplane position
                switch self.hyperplane {
                case .xConstant:
                    self.hyperplane = .xConstant(value)
                case .yConstant:
                    self.hyperplane = .yConstant(value)
                case .zConstant:
                    self.hyperplane = .zConstant(value)
                case .wConstant:
                    self.hyperplane = .wConstant(value)
                case .custom(let normal, _):
                    self.hyperplane = .custom(normal, value)
                }

                // Generate and return the cross-section
                let section = self.generateCrossSection()

                DispatchQueue.main.async {
                    callback(section)
                }

                // Small delay to make the animation visible
                Thread.sleep(forTimeInterval: 0.05)
            }
        }
    }

    // MARK: - GA4D Integration

    /// Generate a cross-section using Geometric Algebra
    /// - Returns: The cross-section representation
    public func generateGACrossSection() -> CrossSection {
        // Step 1: Extract hyperplane normal and distance
        let (normal, distance) = getHyperplaneParameters()

        // Step 2: Use GA to represent the hyperplane
        let normalMV = GA4D.Metric.Multivector.vector(normal)

        // Find hyperplane intersections using GA
        var intersections: [SIMD3<Float>] = []
        var processedEdges = Set<String>()

        for (i, j) in edges4D {
            // Create unique edge ID
            let edgeID = "\(min(i, j))-\(max(i, j))"
            if processedEdges.contains(edgeID) { continue }
            processedEdges.insert(edgeID)

            // Get edge vertices
            let p1 = vertices4D[i]
            let p2 = vertices4D[j]

            // Convert to multivectors
            let mv1 = GA4D.Metric.Multivector.vector(p1)
            let mv2 = GA4D.Metric.Multivector.vector(p2)

            // Calculate dot products with normal (distance from hyperplane)
            let d1 = (GA4D.Metric.GeometricProduct.innerProduct(normalMV, mv1)[0]) - distance
            let d2 = (GA4D.Metric.GeometricProduct.innerProduct(normalMV, mv2)[0]) - distance

            // Check if edge intersects the hyperplane
            if (d1 * d2 <= 0) && (abs(d1) + abs(d2) > 0) {
                // Calculate intersection parameter
                let t = abs(d1) / (abs(d1) + abs(d2))

                // Interpolate to find intersection point
                let intersection4D = mix(p1, p2, t: t)

                // Project to 3D by dropping the coordinate corresponding to largest normal component
                let intersection3D = projectTo3D(intersection4D, normal: normal)

                intersections.append(intersection3D)
            }
        }

        // Construct the cross-section from intersections
        let (sectionVertices, sectionEdges) = constructCrossSectionMesh(from: intersections)

        // Generate colors based on GA properties
        let colors = generateGAColors(for: sectionVertices)

        // Create faces
        let faces = triangulateIntersection(sectionVertices, sectionEdges)

        return CrossSection(
            vertices: sectionVertices,
            edges: sectionEdges,
            faces: faces,
            colors: colors
        )
    }

    /// Generate colors for cross-section vertices based on GA properties
    /// - Parameter vertices: The 3D vertices to color
    /// - Returns: Array of color values
    private func generateGAColors(for vertices: [SIMD3<Float>]) -> [SIMD4<Float>] {
        // This is an enhanced version using GA properties
        return vertices.map { vertex in
            // Create a 4D point with the hyperplane coordinate added back
            let (normal, distance) = getHyperplaneParameters()

            // Create a corresponding 4D point by placing it on the hyperplane
            // Find index of max normal component
            let maxIndex =
                [abs(normal.x), abs(normal.y), abs(normal.z), abs(normal.w)]
                .enumerated()
                .max(by: { $0.element < $1.element })?
                .offset ?? 3

            // Add the w component back based on normal equation of hyperplane
            var point4D = SIMD4<Float>(0, 0, 0, 0)
            var idx = 0

            for i in 0..<4 {
                if i == maxIndex {
                    // Solve for missing coordinate using plane equation
                    // n·p = d -> missing = (d - n·partial)/n_max
                    var partialSum: Float = 0
                    var j = 0
                    for k in 0..<4 {
                        if k != maxIndex {
                            if j < 3 {
                                partialSum += normal[k] * vertex[j]
                                point4D[k] = vertex[j]
                                j += 1
                            }
                        }
                    }
                    point4D[i] = (distance - partialSum) / normal[i]
                } else if idx < 3 {
                    point4D[i] = vertex[idx]
                    idx += 1
                }
            }

            // Convert to multivector
            let mv = GA4D.Metric.Multivector.vector(point4D)

            // Create bivector representing orientation
            let bivector = GA4D.Metric.Multivector.basis("e12", dimension: .dim4)

            // Calculate geometric product to get orientation information
            let product = GA4D.Metric.GeometricProduct.geometricProduct(mv, bivector)

            // Extract scalar and bivector parts for coloring
            let scalar = abs(product[0])
            let bivector1 = abs(product["e12"])
            let bivector2 = abs(product["e23"])

            // Normalize values for colors
            let r = bivector1 / (bivector1 + bivector2 + scalar + 0.001)
            let g = bivector2 / (bivector1 + bivector2 + scalar + 0.001)
            let b = scalar / (bivector1 + bivector2 + scalar + 0.001)

            return SIMD4<Float>(r, g, b, 1.0)
        }
    }

    // MARK: - Helper Methods

    /// Convert hyperplane type to normal and distance
    private func getHyperplaneParameters() -> (SIMD4<Float>, Float) {
        switch hyperplane {
        case .xConstant(let value):
            return (SIMD4<Float>(1, 0, 0, 0), value)

        case .yConstant(let value):
            return (SIMD4<Float>(0, 1, 0, 0), value)

        case .zConstant(let value):
            return (SIMD4<Float>(0, 0, 1, 0), value)

        case .wConstant(let value):
            return (SIMD4<Float>(0, 0, 0, 1), value)

        case .custom(let normal, let distance):
            // Ensure the normal is normalized
            return (normalize(normal), distance)
        }
    }

    /// Find all edge intersections with the hyperplane
    private func findEdgeIntersections(normal: SIMD4<Float>, distance: Float) -> [SIMD3<Float>] {
        var intersections: [SIMD3<Float>] = []
        var edgeProcessed = Set<String>()  // Track processed edges to avoid duplicates

        // For each edge in the 4D object
        for (i, j) in edges4D {
            // Create a unique identifier for this edge
            let edgeId = "\(min(i, j))-\(max(i, j))"
            if edgeProcessed.contains(edgeId) {
                continue
            }
            edgeProcessed.insert(edgeId)

            // Get the edge vertices
            let v1 = vertices4D[i]
            let v2 = vertices4D[j]

            // Calculate signed distances from hyperplane
            let d1 = simd_dot(normal, v1) - distance
            let d2 = simd_dot(normal, v2) - distance

            // Check if edge crosses the hyperplane
            if (d1 * d2 <= 0) && (abs(d1) + abs(d2) > 0) {
                // Calculate interpolation factor
                let t = abs(d1) / (abs(d1) + abs(d2))

                // Interpolate to find intersection point in 4D
                let intersection4D = mix(v1, v2, t: t)

                // Project to 3D by dropping the coordinate corresponding to the largest normal component
                let intersection3D = projectTo3D(intersection4D, normal: normal)

                intersections.append(intersection3D)
            }
        }

        return intersections
    }

    /// Project a 4D point to 3D by dropping the coordinate corresponding to the largest normal component
    private func projectTo3D(_ point4D: SIMD4<Float>, normal: SIMD4<Float>) -> SIMD3<Float> {
        // Find which dimension to project out (largest component of normal)
        let absNormal = abs(normal)
        var maxComponent = 0
        var maxValue: Float = absNormal[0]

        for i in 1..<4 {
            if absNormal[i] > maxValue {
                maxValue = absNormal[i]
                maxComponent = i
            }
        }

        // Construct 3D point by dropping the identified component
        switch maxComponent {
        case 0:
            return SIMD3<Float>(point4D.y, point4D.z, point4D.w)
        case 1:
            return SIMD3<Float>(point4D.x, point4D.z, point4D.w)
        case 2:
            return SIMD3<Float>(point4D.x, point4D.y, point4D.w)
        case 3:
            return SIMD3<Float>(point4D.x, point4D.y, point4D.z)
        default:
            return SIMD3<Float>(point4D.x, point4D.y, point4D.z)  // Default fallback
        }
    }

    /// Construct a cross-section mesh from intersection points
    private func constructCrossSectionMesh(from intersections: [SIMD3<Float>]) -> (
        [SIMD3<Float>], [(Int, Int)]
    ) {
        var uniqueVertices: [SIMD3<Float>] = []
        var edges: [(Int, Int)] = []

        // Filter duplicate vertices based on proximity
        for p in intersections {
            var isDuplicate = false
            for (_, existing) in uniqueVertices.enumerated() {
                if simd_distance(p, existing) < intersectionThreshold {
                    isDuplicate = true
                    break
                }
            }

            if !isDuplicate {
                uniqueVertices.append(p)
            }
        }

        // Connect vertices that are within a reasonable distance
        for i in 0..<uniqueVertices.count {
            for j in (i + 1)..<uniqueVertices.count {
                let distance = simd_distance(uniqueVertices[i], uniqueVertices[j])
                if distance < maxSearchDistance {
                    // Check if this is likely to be an edge (by proximity or other heuristics)
                    edges.append((i, j))
                }
            }
        }

        return (uniqueVertices, edges)
    }

    /// Generate colors for cross-section vertices based on their position
    private func generateColors(for vertices: [SIMD3<Float>]) -> [SIMD4<Float>] {
        return vertices.map { vertex in
            // Create a color based on the position
            let normalizedPos = normalize(vertex)

            // Map from [-1,1] to [0,1] range
            let r = (normalizedPos.x + 1) * 0.5
            let g = (normalizedPos.y + 1) * 0.5
            let b = (normalizedPos.z + 1) * 0.5

            return SIMD4<Float>(r, g, b, 1.0)
        }
    }

    /// Simple triangulation of the cross-section
    private func triangulateIntersection(_ vertices: [SIMD3<Float>], _ edges: [(Int, Int)])
        -> [[Int]]
    {
        var faces: [[Int]] = []

        // Basic triangulation - identify small cycles in the edge graph
        // Note: This is a simplified approach and may not work for all cases
        // A more robust approach would use a proper triangulation algorithm

        // Find all potential triangles
        for i in 0..<edges.count {
            let (a, b) = edges[i]

            for j in (i + 1)..<edges.count {
                let (c, d) = edges[j]

                // Check if edges share a vertex
                if a == c {
                    // Look for an edge connecting b and d
                    if edges.contains(where: { ($0 == b && $1 == d) || ($0 == d && $1 == b) }) {
                        faces.append([a, b, d])
                    }
                } else if a == d {
                    // Look for an edge connecting b and c
                    if edges.contains(where: { ($0 == b && $1 == c) || ($0 == c && $1 == b) }) {
                        faces.append([a, b, c])
                    }
                } else if b == c {
                    // Look for an edge connecting a and d
                    if edges.contains(where: { ($0 == a && $1 == d) || ($0 == d && $1 == a) }) {
                        faces.append([b, a, d])
                    }
                } else if b == d {
                    // Look for an edge connecting a and c
                    if edges.contains(where: { ($0 == a && $1 == c) || ($0 == c && $1 == a) }) {
                        faces.append([b, a, c])
                    }
                }
            }
        }

        return faces
    }
}
