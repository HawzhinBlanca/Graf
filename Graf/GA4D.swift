import Accelerate
import Foundation
import simd

// Operator declarations at file scope - required for Swift 5
infix operator ^ : MultiplicationPrecedence
infix operator • : MultiplicationPrecedence

/// The GA4D namespace contains production-ready implementation of Geometric Algebra
/// optimized for 4D visualization and transformations.
///
/// This implementation focuses on performance, numerical stability, and integration with
/// the Metal renderer while maintaining the mathematical correctness required for
/// Geometric Algebra operations.
public struct GA4D {

    private init() {}

    // MARK: - Type Definitions

    /// Defines the dimension of the vector space
    public enum Dimension: Int, CaseIterable {
        case dim2 = 2
        case dim3 = 3
        case dim4 = 4

        /// Number of basis blades in this dimension (2^n)
        public var bladeCount: Int {
            return 1 << self.rawValue
        }

        /// Number of basis vectors in this dimension
        public var vectorCount: Int {
            return self.rawValue
        }

        /// The pseudoscalar index for this dimension
        public var pseudoscalarIndex: Int {
            return (1 << self.rawValue) - 1
        }
    }

    /// Metric signatures for different geometric algebras
    public enum Metric: Equatable {
        case euclidean  // Signature (+,+,+,+)
        case minkowski  // Signature (-,+,+,+) for spacetime
        case conformal  // Signature (+,+,+,+,-) for conformal GA
        case projective  // Signature (+,+,+,+,0) for projective GA
        case custom(signature: [Float])  // Custom signature

        /// Get the signature array for a specific dimension
        public func signature(for dimension: Dimension) -> [Float] {
            switch self {
            case .euclidean:
                return Array(repeating: Float(1.0), count: dimension.rawValue)
            case .minkowski:
                var sig = Array(repeating: Float(1.0), count: dimension.rawValue)
                sig[0] = Float(-1.0)  // Time dimension has negative signature
                return sig
            case .conformal:
                var sig = Array(repeating: Float(1.0), count: dimension.rawValue + 1)
                sig[dimension.rawValue] = Float(-1.0)  // Extra dimension has negative signature
                return sig
            case .projective:
                var sig = Array(repeating: Float(1.0), count: dimension.rawValue + 1)
                sig[dimension.rawValue] = Float(0.0)  // Extra dimension has zero signature
                return sig
            case .custom(let signature):
                // Ensure we're returning [Float], not [Double]
                return signature.map { Float($0) }
            }
        }

        /// Enumeration for basis blade types (grades)
        public enum Grade: Int, CaseIterable {
            case scalar = 0  // Grade 0 (scalar)
            case vector = 1  // Grade 1 (vector)
            case bivector = 2  // Grade 2 (area)
            case trivector = 3  // Grade 3 (volume)
            case quadvector = 4  // Grade 4 (hypervolume)

            /// Get all grades available for a specific dimension
            public static func all(for dimension: Dimension) -> [Grade] {
                return Grade.allCases.filter { $0.rawValue <= dimension.rawValue }
            }

            /// Get the number of blades of this grade in the given dimension
            public func bladeCount(in dimension: Dimension) -> Int {
                // Binomial coefficient (n choose k) where n is dimension and k is grade
                let n = dimension.rawValue
                let k = self.rawValue

                if k > n { return 0 }
                if k == 0 || k == n { return 1 }

                // Calculate binomial coefficient
                var result = 1
                for i in 1...k {
                    result = result * (n - (k - i)) / i
                }
                return result
            }

            /// Get all blade indices for this grade in the given dimension
            public func bladeIndices(in dimension: Dimension) -> [Int] {
                var result: [Int] = []

                // For each possible blade index
                for i in 0..<dimension.bladeCount {
                    if Multivector.grade(ofBlade: i) == self.rawValue {
                        result.append(i)
                    }
                }

                return result
            }
        }

        // MARK: - Multivector Implementation

        /// A structure representing a multivector in Geometric Algebra.
        /// This is the fundamental data type in GA, representing elements
        /// of all grades (scalars, vectors, bivectors, etc.) in a unified way.
        public struct Multivector: Equatable {
            /// The dimension of the vector space
            public let dimension: Dimension

            /// The metric signature of the algebra
            public let metric: Metric

            /// The components of the multivector stored as a contiguous array
            /// Indexed by blade index (binary representation)
            private var components: [Float]

            public static func getProductTable(key: String) -> ProductTable? {
                return productTables[key]
            }

            static func setProductTable(table: ProductTable, forKey key: String) {
                productTables[key] = table
            }

            /// Cache lookup tables for geometric operations (shared between instances)
            private static var productTables: [String: ProductTable] = [:]

            /// Create a zero multivector of the specified dimension and metric
            public init(dimension: Dimension, metric: Metric = .euclidean) {
                self.dimension = dimension
                self.metric = metric
                self.components = Array(repeating: 0.0, count: dimension.bladeCount)
            }

            /// Create a multivector with specified components
            public init(dimension: Dimension, metric: Metric = .euclidean, components: [Float]) {
                self.dimension = dimension
                self.metric = metric

                // Ensure proper size or pad with zeros
                let bladeCount = dimension.bladeCount
                if components.count == bladeCount {
                    self.components = components
                } else if components.count < bladeCount {
                    self.components =
                        components + Array(repeating: 0.0, count: bladeCount - components.count)
                } else {
                    self.components = Array(components[0..<bladeCount])
                }
            }

            // MARK: - Equatable implementation

            public static func == (lhs: Multivector, rhs: Multivector) -> Bool {
                guard lhs.dimension == rhs.dimension, lhs.metric == rhs.metric else {
                    return false
                }
                return lhs.components == rhs.components
            }

            // MARK: - Static Constructors

            /// Create a scalar multivector
            public static func scalar(
                _ value: Float, dimension: Dimension, metric: Metric = .euclidean
            ) -> Multivector {
                var components = Array(repeating: Float(0.0), count: dimension.bladeCount)
                components[0] = value  // Directly using Float
                return Multivector(dimension: dimension, metric: metric, components: components)
            }

            /// Create a pseudoscalar for a given dimension and metric
            public static func pseudoscalar(dimension: Dimension, metric: Metric = .euclidean)
                -> Multivector
            {
                var result = Multivector(dimension: dimension, metric: metric)

                // Pseudoscalar is the highest-grade basis blade (index 2^dimension - 1)
                result[dimension.pseudoscalarIndex] = 1.0

                return result
            }

            /// Create a basis blade multivector
            public static func basis(index: Int, dimension: Dimension, metric: Metric = .euclidean)
                -> Multivector
            {
                var result = Multivector(dimension: dimension, metric: metric)
                result[index] = 1.0
                return result
            }

            /// Create a basis blade multivector using a string representation like "e12" or "e134"
            public static func basis(
                _ basisString: String, dimension: Dimension, metric: Metric = .euclidean
            ) -> Multivector {
                let index = indexFromBasisString(basisString, dimension: dimension)
                return basis(index: index, dimension: dimension, metric: metric)
            }

            /// Create a vector multivector from components
            public static func vector(
                _ components: [Float], dimension: Dimension, metric: Metric = .euclidean
            ) -> Multivector {
                precondition(
                    components.count <= dimension.rawValue, "Vector components exceed dimension")

                var mvComponents: [Float] = Array(repeating: 0.0, count: dimension.bladeCount)

                // Place vector components at 2^i positions (e1, e2, e3, etc.)
                for i in 0..<min(components.count, dimension.rawValue) {
                    mvComponents[1 << i] = components[i]
                }

                return Multivector(dimension: dimension, metric: metric, components: mvComponents)
            }

            /// Create a vector from SIMD values
            public static func vector(
                _ vector: SIMD2<Float>, dimension: Dimension = .dim2, metric: Metric = .euclidean
            ) -> Multivector {
                return Multivector.vector(
                    [vector.x, vector.y], dimension: dimension, metric: metric)
            }

            public static func vector(
                _ vector: SIMD3<Float>, dimension: Dimension = .dim3, metric: Metric = .euclidean
            ) -> Multivector {
                return Multivector.vector(
                    [vector.x, vector.y, vector.z], dimension: dimension, metric: metric)
            }

            public static func vector(
                _ vector: SIMD4<Float>, dimension: Dimension = .dim4, metric: Metric = .euclidean
            ) -> Multivector {
                return Multivector.vector(
                    [vector.x, vector.y, vector.z, vector.w], dimension: dimension, metric: metric)
            }

            /// Create a bivector multivector
            public static func bivector(
                _ components: [Float], dimension: Dimension, metric: Metric = .euclidean
            ) -> Multivector {
                let bivectorCount = Grade.bivector.bladeCount(in: dimension)
                precondition(
                    components.count <= bivectorCount,
                    "Bivector components exceed count for dimension")

                var mvComponents = Array(repeating: Float(0.0), count: dimension.bladeCount)

                // Generate all bivector basis elements (e12, e13, e14, e23, etc.)
                var bivectorIndex = 0
                for i in 0..<dimension.rawValue {
                    for j in (i + 1)..<dimension.rawValue {
                        if bivectorIndex < components.count {
                            // Calculate binary index for the e_i ∧ e_j basis element
                            let bladeIndex = (1 << i) | (1 << j)
                            mvComponents[bladeIndex] = components[bivectorIndex]
                        }
                        bivectorIndex += 1
                    }
                }

                return Multivector(dimension: dimension, metric: metric, components: mvComponents)
            }

            /// Create a bivector from a plane defined by two vectors
            public static func planeFromVectors(
                _ v1: SIMD4<Float>, _ v2: SIMD4<Float>, dimension: Dimension = .dim4,
                metric: Metric = .euclidean
            ) -> Multivector {
                let mv1 = vector(v1, dimension: dimension, metric: metric)
                let mv2 = vector(v2, dimension: dimension, metric: metric)

                // The outer product gives us the plane
                return GeometricProduct.outerProduct(mv1, mv2)
            }

            /// Create a rotor (a specialized even-grade multivector for rotations)
            public static func rotor(angle: Float, plane: Multivector) -> Multivector {
                precondition(plane.grade() == .bivector, "Rotor plane must be a bivector")

                // Normalize the bivector to ensure it represents a valid plane
                let normalizedPlane = plane.normalized()

                // A rotor is defined as R = cos(θ/2) + sin(θ/2)B
                // where B is a unit bivector
                let halfAngle = angle / 2.0
                let cosComponent = cos(halfAngle)
                let sinComponent = sin(halfAngle)

                // Start with a scalar
                var rotor = Multivector.scalar(
                    cosComponent, dimension: plane.dimension, metric: plane.metric)

                // Add the bivector part (scaled by sin(θ/2))
                for i in 0..<plane.components.count {
                    if plane.components[i] != 0
                        && Multivector.grade(ofBlade: i) == Grade.bivector.rawValue
                    {
                        rotor.components[i] = sinComponent * normalizedPlane.components[i]
                    }
                }

                return rotor
            }

            // MARK: - Accessor Methods

            /// Get a component value by index
            public subscript(index: Int) -> Float {
                get {
                    guard index >= 0 && index < components.count else {
                        return 0.0  // Out of bounds access returns 0
                    }
                    return components[index]
                }
                set {
                    guard index >= 0 && index < components.count else {
                        return  // Ignore out of bounds
                    }
                    components[index] = newValue
                }
            }

            /// Subscript access using basis blade name
            public subscript(basis: String) -> Float {
                get {
                    let index = Multivector.indexFromBasisString(basis, dimension: dimension)
                    return self[index]
                }
                set {
                    let index = Multivector.indexFromBasisString(basis, dimension: dimension)
                    self[index] = newValue
                }
            }

            /// Access all components as an array
            public var allComponents: [Float] {
                return components
            }

            // MARK: - Core Geometric Algebra Operations

            /// Get the grade of this multivector (if it's a homogeneous blade)
            public func grade() -> Grade? {
                // Check if this is a homogeneous multivector (all components are of the same grade)
                var foundGrade: Int? = nil
                let epsilon: Float = 1e-6

                for i in 0..<components.count {
                    if abs(components[i]) > epsilon {
                        let bladeGrade = Multivector.grade(ofBlade: i)

                        if let current = foundGrade {
                            if current != bladeGrade {
                                return nil  // Mixed grades
                            }
                        } else {
                            foundGrade = bladeGrade
                        }
                    }
                }

                if let grade = foundGrade {
                    return Grade(rawValue: grade)
                }

                // If all components are zero, return scalar by convention
                return .scalar
            }

            /// Check if this multivector has a specific grade component
            public func hasGrade(_ grade: Grade) -> Bool {
                for i in 0..<components.count {
                    if abs(components[i]) > 1e-6 && Multivector.grade(ofBlade: i) == grade.rawValue
                    {
                        return true
                    }
                }
                return false
            }

            /// Extract a grade part from this multivector
            public func extractGrade(_ grade: Grade) -> Multivector {
                var result = Multivector(dimension: dimension, metric: metric)

                for i in 0..<components.count {
                    if Multivector.grade(ofBlade: i) == grade.rawValue {
                        result.components[i] = components[i]
                    }
                }

                return result
            }

            /// Get the magnitude (norm) of this multivector
            public func magnitude() -> Float {
                // For a general multivector M, the magnitude is sqrt(<M~M>) where ~ is reversion
                // and <> is scalar part extraction

                let reversed = self.reverse()
                let product = GeometricProduct.geometricProduct(self, reversed)

                // Extract scalar part (component 0)
                let scalarPart = product.components[0]

                // Handle negative scalars that can arise from non-Euclidean metrics
                return sqrt(abs(scalarPart))
            }

            /// Normalize this multivector
            public func normalized() -> Multivector {
                let mag = self.magnitude()
                guard mag > 1e-6 else { return self }

                // Manually scale each component by 1/mag
                let invMag = 1.0 / mag
                var result = Multivector(dimension: dimension, metric: metric)
                for i in 0..<components.count {
                    result[i] = components[i] * invMag
                }
                return result
            }

            /// Reverse a multivector (flip sign of odd-grade components)
            public func reverse() -> Multivector {
                var result = self

                for i in 0..<components.count {
                    let grade = Multivector.grade(ofBlade: i)
                    // For grade k, the reverse has sign (-1)^(k(k-1)/2)
                    // This simplifies to -1 for grades 2 and 3, +1 for grades 0, 1, and 4
                    let gradeFactor = (grade * (grade - 1)) / 2
                    if gradeFactor % 2 == 1 {
                        result.components[i] = -components[i]
                    }
                }

                return result
            }

            /// Get the dual of this multivector
            public func dual() -> Multivector {
                // The dual is calculated as M * I⁻¹ where I is the pseudoscalar
                let pseudoscalar = Multivector.pseudoscalar(dimension: dimension, metric: metric)

                // Optimization: In Euclidean space, I⁻¹ = I / I²
                // In 3D Euclidean space, I² = 1, so I⁻¹ = I (reversion of I)
                // In 4D Euclidean space, I² = 1, so I⁻¹ = I (reversion of I)

                if metric == .euclidean {
                    return GeometricProduct.geometricProduct(self, pseudoscalar.reverse())
                } else {
                    return GeometricProduct.geometricProduct(self, pseudoscalar.inverse())
                }
            }

            /// Get the inverse of this multivector
            public func inverse() -> Multivector {
                // For general multivector M, the inverse is ~M / (M~M)
                // where ~M is the reverse of M

                let reversed = self.reverse()
                let product = GeometricProduct.geometricProduct(self, reversed)

                // Extract scalar part
                let scalarPart = product.components[0]

                // Guard against division by zero
                guard abs(scalarPart) > 1e-6 else {
                    // Return zero multivector if no inverse exists
                    return Multivector(dimension: dimension, metric: metric)
                }

                // Manually scale the reversed multivector by 1/scalarPart
                let invScalarPart = 1.0 / scalarPart
                var result = Multivector(dimension: dimension, metric: metric)
                for i in 0..<reversed.components.count {
                    result[i] = reversed.components[i] * invScalarPart
                }
                return result
            }

            /// Apply this multivector as a versor to transform another multivector
            /// For a versor V and a multivector M, the transformation is V * M * V⁻¹
            public func transform(_ other: Multivector) -> Multivector {
                // Ensure this is a versor (unit-magnitude even/odd-grade)
                let mag = self.magnitude()
                guard abs(mag - 1.0) < 1e-4 else {
                    // Normalize if not already a unit
                    return self.normalized().transform(other)
                }

                // For rotation using a rotor R:
                // - Vector transforms as R * v * R⁻¹ = R * v * ~R (since R is normalized)

                let inverse = self.reverse()  // For normalized versors, reverse = inverse

                // Apply sandwich product: V * M * V⁻¹
                let temp = GeometricProduct.geometricProduct(self, other)
                return GeometricProduct.geometricProduct(temp, inverse)
            }

            /// Convert to a SIMD vector when possible (if grade 1)
            public func toVector() -> SIMD4<Float>? {
                // Extract vector components (basis blades e1, e2, e3, e4)
                // instead of requiring a pure vector, extract the vector components even if other components exist
                var result = SIMD4<Float>(0, 0, 0, 0)

                // Extract vector components (basis blades e1, e2, e3, e4)
                for i in 0..<min(4, dimension.rawValue) {
                    let bladeIndex = 1 << i  // 2^i for e_i basis vector
                    result[i] = components[bladeIndex]
                }

                // Only return nil if there are no vector components at all
                if simd_length(result) < 1e-6 {
                    // If the vector part is nearly zero, try to normalize the result to extract a direction
                    var nonVectorPartFound = false
                    for (index, value) in components.enumerated() {
                        if index > 0 && abs(value) > 1e-6 && !isPowerOfTwo(index) {
                            nonVectorPartFound = true
                            break
                        }
                    }

                    if !nonVectorPartFound {
                        return nil  // No significant components found at all
                    }

                    // Try to extract a direction from the highest-magnitude components
                    var maxMagnitude: Float = 0
                    var highestIndex = 0

                    for (index, value) in components.enumerated() {
                        if abs(value) > maxMagnitude {
                            maxMagnitude = abs(value)
                            highestIndex = index
                        }
                    }

                    // If highest component is not a vector component, create a fallback
                    if !isPowerOfTwo(highestIndex) && highestIndex > 0 {
                        // Use a fallback vector (returning something is better than crashing)
                        return SIMD4<Float>(1, 0, 0, 0)
                    }
                }

                return result
            }

            /// Helper function to check if a number is a power of two (for vector basis elements)
            private func isPowerOfTwo(_ n: Int) -> Bool {
                return n > 0 && (n & (n - 1)) == 0
            }

            // MARK: - String Representation

            public var description: String {
                var parts: [String] = []
                let epsilon: Float = 1e-6

                // Add each non-zero component with its basis blade
                for i in 0..<components.count {
                    if abs(components[i]) > epsilon {
                        let bladeString = Multivector.bladeString(forIndex: i, dimension: dimension)
                        let valueString = String(format: "%.4f", components[i])

                        // Special case for scalar (don't need a basis symbol)
                        if i == 0 {
                            parts.append(valueString)
                        } else if abs(components[i] - 1.0) < epsilon {
                            parts.append(bladeString)
                        } else if abs(components[i] + 1.0) < epsilon {
                            parts.append("-\(bladeString)")
                        } else {
                            parts.append("\(valueString)\(bladeString)")
                        }
                    }
                }

                if parts.isEmpty {
                    return "0"  // Zero multivector
                }

                return parts.joined(separator: " + ").replacingOccurrences(of: "+ -", with: "- ")
            }

            // MARK: - Utility Methods

            /// Calculate the grade of a blade from its binary index
            /// The grade is the number of 1 bits in the binary representation
            public static func grade(ofBlade index: Int) -> Int {
                // Count the number of 1s in the binary representation (Hamming weight)
                var temp = index
                var count = 0

                while temp != 0 {
                    temp &= (temp - 1)  // Clear the least significant bit
                    count += 1
                }

                return count
            }

            /// Get string representation of a basis blade
            public static func bladeString(forIndex index: Int, dimension: Dimension) -> String {
                if index == 0 {
                    return "1"  // Scalar
                }

                var result = ""

                // Build string like "e123" for basis blade
                for i in 0..<dimension.rawValue {
                    // Check if the i-th bit is set
                    if (index & (1 << i)) != 0 {
                        result += "\(i+1)"  // Add 1-based index
                    }
                }

                return "e\(result)"
            }

            /// Convert a basis string like "e12" to its binary index
            public static func indexFromBasisString(_ basis: String, dimension: Dimension) -> Int {
                // Handle scalar (1) case
                if basis == "1" || basis.lowercased() == "scalar" {
                    return 0
                }

                // Strip "e" prefix if present
                var sanitized = basis
                if sanitized.hasPrefix("e") {
                    sanitized.removeFirst()
                }

                var index = 0
                for char in sanitized {
                    if let digit = Int(String(char)) {
                        // Ensure digit is valid for dimension (1-based index)
                        guard digit >= 1 && digit <= dimension.rawValue else {
                            continue
                        }

                        // Set the appropriate bit (convert from 1-based to 0-based)
                        index |= (1 << (digit - 1))
                    }
                }

                return index
            }

            // MARK: - Operator Implementations

            // Negation
            public static prefix func - (a: Multivector) -> Multivector {
                var result = a
                for i in 0..<a.allComponents.count {
                    result[i] = -a[i]
                }
                return result
            }

            // Addition
            public static func + (lhs: Multivector, rhs: Multivector) -> Multivector {
                precondition(
                    lhs.dimension == rhs.dimension,
                    "Cannot add multivectors of different dimensions")

                var result = Multivector(dimension: lhs.dimension, metric: lhs.metric)
                for i in 0..<lhs.allComponents.count {
                    result[i] = lhs[i] + rhs[i]
                }
                return result
            }

            // Subtraction
            public static func - (lhs: Multivector, rhs: Multivector) -> Multivector {
                precondition(
                    lhs.dimension == rhs.dimension,
                    "Cannot subtract multivectors of different dimensions")

                var result = Multivector(dimension: lhs.dimension, metric: lhs.metric)
                for i in 0..<lhs.allComponents.count {
                    result[i] = lhs[i] - rhs[i]
                }
                return result
            }

            // Scalar multiplication
            public static func * (lhs: Float, rhs: Multivector) -> Multivector {
                var result = rhs
                for i in 0..<rhs.allComponents.count {
                    result[i] = lhs * rhs[i]
                }
                return result
            }

            public static func * (lhs: Multivector, rhs: Float) -> Multivector {
                return rhs * lhs
            }

            // Geometric product between multivectors
            public static func * (lhs: Multivector, rhs: Multivector) -> Multivector {
                return GeometricProduct.geometricProduct(lhs, rhs)
            }

            // Division by scalar
            public static func / (lhs: Multivector, rhs: Float) -> Multivector {
                precondition(abs(rhs) > 1e-10, "Division by near-zero scalar")

                var result = lhs
                let invRhs = 1.0 / rhs
                for i in 0..<lhs.allComponents.count {
                    result[i] = lhs[i] * invRhs
                }
                return result
            }

            // Wedge product
            public static func ^ (lhs: Multivector, rhs: Multivector) -> Multivector {
                return GeometricProduct.outerProduct(lhs, rhs)
            }

            // Inner product
            public static func • (lhs: Multivector, rhs: Multivector) -> Multivector {
                return GeometricProduct.innerProduct(lhs, rhs)
            }
        }

        // MARK: - Geometric Product Class

        /// Static methods for computing different geometric algebra products
        public struct GeometricProduct {
            /// Compute the geometric product of two multivectors
            public static func geometricProduct(_ a: Multivector, _ b: Multivector) -> Multivector {
                precondition(
                    a.dimension == b.dimension,
                    "Cannot multiply multivectors of different dimensions")

                // Get shared product table key
                let tableKey = "\(a.dimension.rawValue)_\(a.metric)"

                // Check for cached product table
                let table: ProductTable
                if let existingTable = Multivector.getProductTable(key: tableKey) {
                    table = existingTable
                } else {
                    // Create and cache new product table
                    let newTable = ProductTable(dimension: a.dimension, metric: a.metric)
                    Multivector.setProductTable(table: newTable, forKey: tableKey)
                    table = newTable
                }

                var result = Multivector(dimension: a.dimension, metric: a.metric)

                // For each possible output blade
                for c in 0..<a.dimension.bladeCount {
                    var sum: Float = 0.0

                    // For each possible pair of input blades
                    for aIdx in 0..<a.dimension.bladeCount {
                        let aValue = a[aIdx]
                        if abs(aValue) < 1e-6 { continue }  // Skip zero components

                        for bIdx in 0..<b.dimension.bladeCount {
                            let bValue = b[bIdx]
                            if abs(bValue) < 1e-6 { continue }  // Skip zero components

                            // Get cached result using product table
                            let (resultIndex, sign) = table.geometricProduct[aIdx][bIdx]

                            // If this term contributes to the current output blade
                            if resultIndex == c {
                                sum += sign * aValue * bValue
                            }
                        }
                    }

                    // Only set non-zero components
                    if abs(sum) > 1e-10 {
                        result[c] = sum
                    }
                }

                return result
            }

            /// Compute the outer (wedge) product of two multivectors
            public static func outerProduct(_ a: Multivector, _ b: Multivector) -> Multivector {
                precondition(
                    a.dimension == b.dimension,
                    "Cannot apply wedge product to multivectors of different dimensions")

                // Get shared product table key
                let tableKey = "\(a.dimension.rawValue)_\(a.metric)"

                // Check for cached product table
                let table: ProductTable
                if let existingTable = Multivector.getProductTable(key: tableKey) {
                    table = existingTable
                } else {
                    // Create and cache new product table
                    let newTable = ProductTable(dimension: a.dimension, metric: a.metric)
                    Multivector.setProductTable(table: newTable, forKey: tableKey)
                    table = newTable
                }

                var result = Multivector(dimension: a.dimension, metric: a.metric)

                // For each possible output blade
                for c in 0..<a.dimension.bladeCount {
                    var sum: Float = 0.0

                    // For each possible pair of input blades
                    for aIdx in 0..<a.dimension.bladeCount {
                        let aValue = a[aIdx]
                        if abs(aValue) < 1e-6 { continue }  // Skip zero components

                        for bIdx in 0..<b.dimension.bladeCount {
                            let bValue = b[bIdx]
                            if abs(bValue) < 1e-6 { continue }  // Skip zero components

                            // Get cached result using product table
                            if let product = table.outerProduct[aIdx][bIdx], product.0 == c {
                                sum += product.1 * aValue * bValue
                            }
                        }
                    }

                    // Only set non-zero components
                    if abs(sum) > 1e-10 {
                        result[c] = sum
                    }
                }

                return result
            }

            /// Compute the inner product of two multivectors
            public static func innerProduct(_ a: Multivector, _ b: Multivector) -> Multivector {
                precondition(
                    a.dimension == b.dimension,
                    "Cannot apply inner product to multivectors of different dimensions")

                // Get shared product table key
                let tableKey = "\(a.dimension.rawValue)_\(a.metric)"

                // Check for cached product table
                let table: ProductTable
                if let existingTable = Multivector.getProductTable(key: tableKey) {
                    table = existingTable
                } else {
                    // Create and cache new product table
                    let newTable = ProductTable(dimension: a.dimension, metric: a.metric)
                    Multivector.setProductTable(table: newTable, forKey: tableKey)
                    table = newTable
                }

                var result = Multivector(dimension: a.dimension, metric: a.metric)

                // For each possible output blade
                for c in 0..<a.dimension.bladeCount {
                    var sum: Float = 0.0

                    // For each possible pair of input blades
                    for aIdx in 0..<a.dimension.bladeCount {
                        let aValue = a[aIdx]
                        if abs(aValue) < 1e-6 { continue }  // Skip zero components

                        for bIdx in 0..<b.dimension.bladeCount {
                            let bValue = b[bIdx]
                            if abs(bValue) < 1e-6 { continue }  // Skip zero components

                            // Get cached result using product table
                            if let product = table.innerProduct[aIdx][bIdx], product.0 == c {
                                sum += product.1 * aValue * bValue
                            }
                        }
                    }

                    // Only set non-zero components
                    if abs(sum) > 1e-10 {
                        result[c] = sum
                    }
                }

                return result
            }
        }

        // MARK: - Optimized Product Tables

        /// Structure to cache the results of geometric products for improved performance
        public struct ProductTable {
            /// The geometric product table: result[a][b] = (resultBlade, sign)
            let geometricProduct: [[(Int, Float)]]

            /// The outer product table: result[a][b] = (resultBlade, sign) or nil if zero
            let outerProduct: [[(Int, Float)?]]

            /// The inner product table: result[a][b] = (resultBlade, sign) or nil if zero
            let innerProduct: [[(Int, Float)?]]

            /// Create product lookup tables for the given dimension and metric
            init(dimension: Dimension, metric: Metric) {
                let bladeCount = dimension.bladeCount
                let signature = metric.signature(for: dimension)

                // Initialize with empty arrays
                var geometricProduct = Array(
                    repeating: Array(repeating: (0, Float(0.0)), count: bladeCount),
                    count: bladeCount)
                var outerProduct = Array(
                    repeating: [(Int, Float)?](repeating: nil, count: bladeCount), count: bladeCount
                )
                var innerProduct = Array(
                    repeating: [(Int, Float)?](repeating: nil, count: bladeCount), count: bladeCount
                )

                // Precompute all possible products
                for a in 0..<bladeCount {
                    for b in 0..<bladeCount {
                        // Geometric product
                        geometricProduct[a][b] = ProductTable.computeGeometricProduct(
                            a, b, dimension, signature)
                        // Outer product
                        if (a & b) == 0 {  // No common basis vectors
                            let aGrade = Multivector.grade(ofBlade: a)
                            let bGrade = Multivector.grade(ofBlade: b)
                            let _ = aGrade + bGrade
                            let expectedBlade = a | b

                            // Calculate sign
                            var sign: Float = 1.0
                            var inversions = 0

                            // Count inversions between basis vectors of a and b
                            for i in 0..<dimension.rawValue {
                                if (a & (1 << i)) != 0 {
                                    for j in 0..<i {
                                        if (b & (1 << j)) != 0 {
                                            inversions += 1
                                        }
                                    }
                                }
                            }

                            if inversions % 2 == 1 {
                                sign = -1.0
                            }

                            outerProduct[a][b] = (expectedBlade, sign)
                        }

                        // Inner product
                        let aGrade = Multivector.grade(ofBlade: a)
                        let bGrade = Multivector.grade(ofBlade: b)
                        let resultGrade = abs(aGrade - bGrade)

                        // Compute using geometric product and filter by grade
                        let (resultBlade, sign) = geometricProduct[a][b]
                        if Multivector.grade(ofBlade: resultBlade) == resultGrade {
                            innerProduct[a][b] = (resultBlade, sign)
                        }
                    }
                }

                self.geometricProduct = geometricProduct
                self.outerProduct = outerProduct
                self.innerProduct = innerProduct
            }

            /// Helper to compute a geometric product term between two basis blades
            private static func computeGeometricProduct(
                _ a: Int, _ b: Int, _ dimension: Dimension, _ signature: [Float]
            ) -> (Int, Float) {
                // Determine which basis vectors are in both a and b
                let commonBits = a & b

                // Calculate the resulting blade (XOR for vectors that don't commute)
                let resultBlade = a ^ b

                // Calculate sign using the reordering formula
                var sign: Float = 1.0

                // For each bit in a
                for i in 0..<dimension.rawValue {
                    if (a & (1 << i)) != 0 {
                        // Count how many bits in b are before this bit
                        for j in 0..<i {
                            if (b & (1 << j)) != 0 {
                                sign *= -1.0  // Anticommutation
                            }
                        }
                    }
                }

                // Apply metric for common bits (which square to +/-1)
                if commonBits != 0 {
                    for i in 0..<dimension.rawValue {
                        if (commonBits & (1 << i)) != 0 && i < signature.count {
                            sign *= signature[i]
                        }
                    }
                }

                return (resultBlade, sign)
            }
        }

        // MARK: - 4D Operations

        /// Specialized 4D transformations using Geometric Algebra
        public struct Operations4D {
            /// Creates a rotor representing a rotation in an arbitrary plane
            /// - Parameters:
            ///   - angle: The angle of rotation in radians
            ///   - planeType: Definition of the plane (e.g., "e12", "e13", "e14", etc.)
            /// - Returns: A rotor multivector representing the rotation
            public static func rotationInPlane(angle: Float, planeType: String) -> Multivector {
                // Create the bivector for the specified plane
                var bivector = Multivector(dimension: .dim4)
                bivector[planeType] = 1.0

                // Create the rotor
                return Multivector.rotor(angle: angle, plane: bivector)
            }

            /// Standard 4D rotation in XY plane
            public static func rotationXY(angle: Float) -> Multivector {
                return rotationInPlane(angle: angle, planeType: "e12")
            }

            /// Standard 4D rotation in XZ plane
            public static func rotationXZ(angle: Float) -> Multivector {
                return rotationInPlane(angle: angle, planeType: "e13")
            }

            /// Standard 4D rotation in XW plane
            public static func rotationXW(angle: Float) -> Multivector {
                return rotationInPlane(angle: angle, planeType: "e14")
            }

            /// Standard 4D rotation in YZ plane
            public static func rotationYZ(angle: Float) -> Multivector {
                return rotationInPlane(angle: angle, planeType: "e23")
            }

            /// Standard 4D rotation in YW plane
            public static func rotationYW(angle: Float) -> Multivector {
                return rotationInPlane(angle: angle, planeType: "e24")
            }

            /// Standard 4D rotation in ZW plane
            public static func rotationZW(angle: Float) -> Multivector {
                return rotationInPlane(angle: angle, planeType: "e34")
            }

            /// Rotate a 4D vector using a rotor
            /// - Parameters:
            ///   - vector: The 4D vector to rotate
            ///   - rotor: The rotor to apply
            /// - Returns: The rotated 4D vector
            public static func rotate(vector: SIMD4<Float>, using rotor: Multivector) -> SIMD4<
                Float
            > {
                // Convert vector to multivector
                let mvVector = Multivector.vector(vector)

                // Apply sandwich product R * v * R⁻¹
                let transformedMV = rotor.transform(mvVector)

                // Convert back to SIMD4, falling back to the original vector if conversion fails
                guard let result = transformedMV.toVector() else {
                    print(
                        "Warning: Could not convert multivector back to vector - using fallback approach"
                    )

                    // Since we can't access components directly, create a new fallback vector
                    // using the original vector's direction but preserving magnitude
                    let originalMagnitude = simd_length(vector)
                    let normalizedOriginal =
                        vector / (originalMagnitude > 0 ? originalMagnitude : 1.0)

                    // We can use the magnitude of the transformed multivector as a hint
                    let transformedMagnitude = transformedMV.magnitude()

                    return normalizedOriginal * transformedMagnitude
                }

                return result
            }

            /// Creates a rotor that rotates from one direction to another
            /// - Parameters:
            ///   - from: The starting direction vector
            ///   - to: The target direction vector
            /// - Returns: A rotor that rotates from the 'from' direction to the 'to' direction
            public static func rotorBetweenVectors(from: SIMD4<Float>, to: SIMD4<Float>)
                -> Multivector
            {
                // Normalize input vectors
                let normalizedFrom = normalize(from)
                let normalizedTo = normalize(to)

                // Convert vectors to multivectors
                let a = Multivector.vector(normalizedFrom)
                let b = Multivector.vector(normalizedTo)

                // The rotor R that rotates a to b can be computed as:
                // R = sqrt(ba) = (1 + ba) / |1 + ba|
                // where ba is the geometric product of b and a

                // Compute the geometric product
                let ba = GeometricProduct.geometricProduct(b, a)

                // Extract the scalar part (dot product)
                let dotProduct = ba[0]

                // If vectors are nearly parallel, return identity rotor
                if abs(dotProduct - 1.0) < 1e-6 {
                    return Multivector.scalar(1.0, dimension: .dim4)
                }

                // If vectors are nearly opposite, create a perpendicular vector and rotate around it
                if abs(dotProduct + 1.0) < 1e-6 {
                    // Find a vector perpendicular to 'from'
                    var perp = SIMD4<Float>(0, 0, 1, 0)
                    if abs(dot(normalizedFrom, perp)) > 0.9 {
                        perp = SIMD4<Float>(0, 1, 0, 0)
                    }

                    // Create a bivector representing the rotation plane
                    let perpMV = Multivector.vector(perp)
                    let bivector = GeometricProduct.outerProduct(a, perpMV).normalized()

                    // Return a 180° rotation
                    return Multivector.rotor(angle: .pi, plane: bivector)
                }

                // Create a rotor that aligns the vectors
                // We use (1 + ba) and normalize to avoid numerical issues with sqrt(ba)
                // Instead of 1 + ba, manually create the combined multivector
                let one = Multivector.scalar(1.0, dimension: ba.dimension, metric: ba.metric)
                var combinedRotor = one
                for i in 0..<ba.dimension.bladeCount {
                    combinedRotor[i] += ba[i]
                }

                return combinedRotor.normalized()
            }

            /// Apply a sequence of rotors to a vector
            /// - Parameters:
            ///   - vector: The vector to transform
            ///   - rotors: An array of rotors to apply in sequence
            /// - Returns: The transformed vector
            public static func applyRotorSequence(to vector: SIMD4<Float>, rotors: [Multivector])
                -> SIMD4<Float>
            {
                // Combine all rotors into a single rotor
                var combinedRotor: Multivector

                if rotors.isEmpty {
                    return vector
                } else if rotors.count == 1 {
                    combinedRotor = rotors[0]
                } else {
                    combinedRotor = rotors[0]

                    for i in 1..<rotors.count {
                        combinedRotor = GeometricProduct.geometricProduct(combinedRotor, rotors[i])
                    }
                }

                // Normalize to ensure we have a valid rotor
                combinedRotor = combinedRotor.normalized()

                // Apply the combined rotor
                return rotate(vector: vector, using: combinedRotor)
            }

            /// Creates a 4D reflection versor
            /// - Parameter normal: The normal vector to the reflection hyperplane
            /// - Returns: A versor representing the reflection
            public static func reflection(normal: SIMD4<Float>) -> Multivector {
                // In GA, a reflection is represented by the normal vector itself
                // The reflection of a vector v is given by -n*v*n where n is normalized
                return Multivector.vector(normalize(normal))
            }

            /// Reflect a vector in a hyperplane
            /// - Parameters:
            ///   - vector: The vector to reflect
            ///   - normal: The normal to the reflection hyperplane
            /// - Returns: The reflected vector
            public static func reflect(vector: SIMD4<Float>, in normal: SIMD4<Float>) -> SIMD4<
                Float
            > {
                // Create reflection versor
                let reflector = reflection(normal: normal)

                // Convert vector to multivector
                let mvVector = Multivector.vector(vector)

                // Apply reflection using the sandwich product but with negative sign
                // Instead of -n*v*n, we'll compute n*v*n and negate the result
                let n = reflector
                let temp1 = GeometricProduct.geometricProduct(n, mvVector)
                let temp2 = GeometricProduct.geometricProduct(temp1, n)

                // Create negated result by flipping sign of each component
                var negatedResult = Multivector(dimension: temp2.dimension, metric: temp2.metric)
                for i in 0..<temp2.dimension.bladeCount {
                    negatedResult[i] = -temp2[i]
                }

                // Convert back to SIMD4
                guard let reflected = negatedResult.toVector() else {
                    fatalError("Failed to convert result back to vector")
                }

                return reflected
            }

            /// Generate a 4D stereographic projection matrix using geometric algebra
            /// - Returns: A projection matrix
            public static func stereographicProjectionMatrix() -> simd_float4x4 {
                // Initialize with identity
                var matrix = simd_float4x4(diagonal: SIMD4<Float>(repeating: 1.0))

                // This is a non-linear projection, but we can create a matrix that
                // applies the projection for points near the origin

                // Using 4D stereographic projection from the "north pole" (0,0,0,1)
                // For a point (x,y,z,w), the projection is (x',y',z') where:
                // x' = x/(1-w), y' = y/(1-w), z' = z/(1-w)

                // The matrix approximates this for small values of w
                matrix[0][3] = -1.0  // x adjustment: x/(1-w) ≈ x(1+w) for small w
                matrix[1][3] = -1.0  // y adjustment
                matrix[2][3] = -1.0  // z adjustment
                matrix[3][3] = 0.0  // Zero out w in result

                return matrix
            }

            /// Create a translator (for translations in conformal or projective GA)
            public static func translator(direction: SIMD3<Float>, distance: Float) -> Multivector {
                // In conformal GA, a translator is e^(t*d∧n) where:
                // - d is direction vector
                // - n is infinity vector (typically e4 in a 3D conformal model)
                // - t is distance

                // Create a conformal model in 4D
                let dimension: Dimension = .dim4
                let metric: Metric = .conformal

                // Create direction vector (embedded in conformal model)
                let dirVector = Multivector.vector(
                    [direction.x, direction.y, direction.z, 0.0],
                    dimension: dimension,
                    metric: metric)

                // Get the infinity basis vector (e4 in this model)
                let infinityVector = Multivector.basis("e4", dimension: dimension, metric: metric)

                // Create the bivector representing direction wedge infinity
                let translationBivector = GeometricProduct.outerProduct(dirVector, infinityVector)

                // Scale by distance
                var scaledBivector = Multivector(dimension: dimension, metric: metric)
                for i in 0..<dimension.bladeCount {
                    scaledBivector[i] = translationBivector[i] * distance
                }

                // Exponentiate using Taylor series approximation
                // For small translations: exp(T) ≈ 1 + T + T²/2! + T³/3! + ...
                // We'll use up to second order for efficiency
                let one = Multivector.scalar(1.0, dimension: dimension, metric: metric)
                let firstOrder = scaledBivector
                let secondOrderTerm = GeometricProduct.geometricProduct(
                    scaledBivector, scaledBivector)

                // Create the T²/2 term by manually scaling secondOrderTerm by 0.5
                var secondOrder = Multivector(dimension: dimension, metric: metric)
                for i in 0..<dimension.bladeCount {
                    secondOrder[i] = secondOrderTerm[i] * 0.5
                }

                // Calculate 1 + T
                var result = one
                for i in 0..<dimension.bladeCount {
                    result[i] += firstOrder[i]
                }

                // Add T²/2 to get final result
                for i in 0..<dimension.bladeCount {
                    result[i] += secondOrder[i]
                }

                return result.normalized()
            }
        }

        // MARK: - 4D Primitives

        /// Generator for 4D primitive shapes using GA
        public struct Primitives {

            /// Creates vertices for a tesseract (4D hypercube) using GA
            /// - Parameter scale: The scale factor for the tesseract
            /// - Returns: Array of 4D vertices
            public static func createTesseract(scale: Float = 1.0) -> [SIMD4<Float>] {
                var vertices: [SIMD4<Float>] = []

                // Generate all 16 vertices of the tesseract
                for w in [0, 1] {
                    for z in [0, 1] {
                        for y in [0, 1] {
                            for x in [0, 1] {
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

                return vertices
            }

            /// Create edges for a tesseract
            /// - Returns: Array of index pairs defining edges
            public static func createTesseractEdges() -> [(Int, Int)] {
                var edges: [(Int, Int)] = []
                let vertices = createTesseract()

                // Generate edges connecting vertices that differ by exactly one coordinate
                for i in 0..<vertices.count {
                    for j in (i + 1)..<vertices.count {
                        // Count differing coordinates
                        let v1 = vertices[i]
                        let v2 = vertices[j]

                        var diffCount = 0
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

                return edges
            }

            /// Creates a 4D hypersphere using GA
            /// - Parameters:
            ///   - radius: The radius of the hypersphere
            ///   - resolution: The resolution (number of subdivisions)
            /// - Returns: Array of 4D vertices
            public static func createHypersphere(radius: Float = 1.0, resolution: Int = 32)
                -> [SIMD4<Float>]
            {
                var vertices: [SIMD4<Float>] = []

                // Generate points using 4D spherical coordinates
                let stepTheta = 2.0 * Float.pi / Float(resolution)
                let stepPhi = Float.pi / Float(resolution)
                let stepPsi = Float.pi / Float(resolution)

                for i in 0...resolution {
                    let theta = Float(i) * stepTheta

                    for j in 0...resolution {
                        let phi = Float(j) * stepPhi

                        for k in 0...resolution / 4 {  // Reduced resolution in 4th dimension for efficiency
                            let psi = Float(k) * stepPsi

                            // Calculate 4D spherical coordinates using GA rotations
                            // Start with a point on the w-axis
                            let basePoint = SIMD4<Float>(0, 0, 0, radius)

                            // Create rotors for each spherical angle
                            let rotorXY = Operations4D.rotationXY(angle: theta)
                            let rotorXZ = Operations4D.rotationXZ(angle: phi)
                            let rotorXW = Operations4D.rotationXW(angle: psi)

                            // Apply rotors in sequence
                            let position = Operations4D.applyRotorSequence(
                                to: basePoint,
                                rotors: [rotorXW, rotorXZ, rotorXY]
                            )

                            vertices.append(position)
                        }
                    }
                }

                return vertices
            }

            /// Creates a Clifford torus using GA
            /// - Parameters:
            ///   - majorRadius: The major radius
            ///   - minorRadius: The minor radius
            ///   - resolution: The resolution
            /// - Returns: Array of 4D vertices
            public static func createCliffordTorus(
                majorRadius: Float = 1.0, minorRadius: Float = 1.0, resolution: Int = 32
            ) -> [SIMD4<Float>] {
                var vertices: [SIMD4<Float>] = []

                // A Clifford torus is a product of two circles in 4D
                let stepU = 2.0 * Float.pi / Float(resolution)
                let stepV = 2.0 * Float.pi / Float(resolution)

                for i in 0...resolution {
                    let u = Float(i) * stepU

                    for j in 0...resolution {
                        let v = Float(j) * stepV

                        // Parameterize using two circles in perpendicular planes
                        // Using GA, we can describe this as two rotations in perpendicular planes

                        // Create circle in XY plane
                        let xyRotor = Operations4D.rotationXY(angle: u)
                        let xyPoint = Operations4D.rotate(
                            vector: SIMD4<Float>(majorRadius, 0, 0, 0),
                            using: xyRotor
                        )

                        // Create circle in ZW plane
                        let zwRotor = Operations4D.rotationZW(angle: v)
                        let zwPoint = Operations4D.rotate(
                            vector: SIMD4<Float>(0, 0, minorRadius, 0),
                            using: zwRotor
                        )

                        // Sum the points to get the Clifford torus point
                        let position = xyPoint + zwPoint

                        vertices.append(position)
                    }
                }

                return vertices
            }
        }
    }

    // MARK: - Metal Integration

    /// Bridge between GA4D and Metal renderer for visualization integration
    public class GA4DMetalBridge {

        /// Transform 4D vertices using GA rotors
        /// - Parameters:
        ///   - vertices: Array of 4D vertices to transform
        ///   - rotations: Rotation angles in radians for each of the 6 planes (xy, xz, xw, yz, yw, zw)
        ///   - projectionType: Type of 4D to 3D projection to apply
        /// - Returns: Array of transformed 3D vertices for rendering
        public static func transformVertices(
            vertices4D: [SIMD4<Float>],
            rotations: (xy: Float, xz: Float, xw: Float, yz: Float, yw: Float, zw: Float),
            projectionType: ProjectionType
        ) -> [SIMD3<Float>] {
            // Create rotors for each plane
            let rotorXY = GA4D.Metric.Operations4D.rotationXY(angle: rotations.xy)
            let rotorXZ = GA4D.Metric.Operations4D.rotationXZ(angle: rotations.xz)
            let rotorXW = GA4D.Metric.Operations4D.rotationXW(angle: rotations.xw)
            let rotorYZ = GA4D.Metric.Operations4D.rotationYZ(angle: rotations.yz)
            let rotorYW = GA4D.Metric.Operations4D.rotationYW(angle: rotations.yw)
            let rotorZW = GA4D.Metric.Operations4D.rotationZW(angle: rotations.zw)

            // Combine rotors (order matters in GA)
            // Apply in order: XY, XZ, XW, YZ, YW, ZW
            let rotors = [rotorXY, rotorXZ, rotorXW, rotorYZ, rotorYW, rotorZW]

            // Array to store projected 3D vertices
            var vertices3D: [SIMD3<Float>] = []

            // Apply rotor to each vertex and project to 3D
            for vertex4D in vertices4D {
                // Apply the combined rotors
                let rotated4D = GA4D.Metric.Operations4D.applyRotorSequence(
                    to: vertex4D, rotors: rotors)

                // Project to 3D
                let projected3D: SIMD3<Float>

                switch projectionType {
                case .stereographic:
                    // Stereographic projection from 4D to 3D
                    let factor: Float = 1.0 / (1.0 - rotated4D.w * 0.1)
                    projected3D = SIMD3<Float>(rotated4D.x, rotated4D.y, rotated4D.z) * factor

                case .perspective:
                    // Perspective projection from 4D to 3D
                    let distance: Float = 5.0
                    let factor = distance / (distance + rotated4D.w)
                    projected3D = SIMD3<Float>(rotated4D.x, rotated4D.y, rotated4D.z) * factor

                case .orthographic:
                    // Orthographic projection from 4D to 3D (simply drop the w coordinate)
                    projected3D = SIMD3<Float>(rotated4D.x, rotated4D.y, rotated4D.z)
                }

                vertices3D.append(projected3D)
            }

            return vertices3D
        }

        /// Calculate vertex normals in 4D space using GA
        /// - Parameter vertices: Array of 4D vertex positions
        /// - Returns: Array of 4D normals corresponding to each vertex
        public static func calculateNormals(
            for vertices4D: [SIMD4<Float>], connections: [(Int, Int, Int)]
        ) -> [SIMD4<Float>] {
            var normals4D: [SIMD4<Float>] = Array(
                repeating: SIMD4<Float>(0, 0, 0, 0), count: vertices4D.count)

            // For each triangular face
            for (i, j, k) in connections {
                guard i < vertices4D.count, j < vertices4D.count, k < vertices4D.count else {
                    continue
                }

                // Extract vertices of the triangle
                let v1 = vertices4D[i]
                let v2 = vertices4D[j]
                let v3 = vertices4D[k]

                // Calculate edge vectors
                let edge1 = v2 - v1
                let edge2 = v3 - v1

                // Create multivectors for edges
                let mv1 = GA4D.Metric.Multivector.vector(edge1)
                let mv2 = GA4D.Metric.Multivector.vector(edge2)

                // Calculate bivector representing the face plane
                let faceElement = GA4D.Metric.GeometricProduct.outerProduct(mv1, mv2)

                // Convert to pseudovector representation of the normal
                let normalMV = faceElement.dual()

                // Convert back to a vector
                guard let normal4D = normalMV.toVector() else {
                    continue
                }

                // Add to each vertex normal (will be normalized later)
                normals4D[i] += normal4D
                normals4D[j] += normal4D
                normals4D[k] += normal4D
            }

            // Normalize all normals
            for i in 0..<normals4D.count {
                let length = simd_length(normals4D[i])
                if length > 1e-6 {
                    normals4D[i] = normals4D[i] / length
                } else {
                    // If no face normal was calculated, use a default
                    normals4D[i] = normalize(vertices4D[i])
                }
            }

            return normals4D
        }

        /// Projection types for 4D to 3D conversion
        public enum ProjectionType {
            case stereographic
            case perspective
            case orthographic
        }
    }
}
