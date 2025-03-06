//
//  that.swift
//  Graf
//
//  Created by HAWZHIN on 05/03/2025.
//

import Foundation
import simd

// Import GA4D if it's a separate module
// If GA4D is not a separate module, then we need to use the full namespace

/// Test and example class that demonstrates the GA4D functionality
class GA4DExamples {

    // MARK: - Basic Multivector Operations

    /// Test basic multivector operations
    static func testBasicOperations() {
        print("Testing basic GA4D operations...")

        // Create a scalar
        let scalar = GA4D.Metric.Multivector.scalar(2.5, dimension: GA4D.Dimension.dim4)
        print("Scalar: \(scalar)")

        // Create a vector
        let vector = GA4D.Metric.Multivector.vector(
            [1.0, 2.0, 3.0, 4.0], dimension: GA4D.Dimension.dim4)
        print("Vector: \(vector)")

        // Create a bivector representing the XY plane
        let bivectorXY = GA4D.Metric.Multivector.basis("e12", dimension: GA4D.Dimension.dim4)
        print("Bivector (XY plane): \(bivectorXY)")

        // Add a scalar and a vector
        let addition = scalar + vector
        print("Scalar + Vector: \(addition)")

        // Compute the geometric product
        let geometricProduct = vector * bivectorXY
        print("Vector * Bivector (geometric product): \(geometricProduct)")

        // Compute the outer product
        let outerProduct = vector ^ bivectorXY
        print("Vector ^ Bivector (outer product): \(outerProduct)")

        // Compute the inner product
        let innerProduct = vector • bivectorXY
        print("Vector • Bivector (inner product): \(innerProduct)")

        // Get the reverse of a multivector
        let reversed = geometricProduct.reverse()
        print("Reversed multivector: \(reversed)")

        // Get the dual of a multivector
        let dual = geometricProduct.dual()
        print("Dual of multivector: \(dual)")

        // Get the magnitude
        let magnitude = vector.magnitude()
        print("Magnitude of vector: \(magnitude)")

        // Normalize
        let normalized = vector.normalized()
        print("Normalized vector: \(normalized)")
    }

    // MARK: - 4D Rotations

    /// Test 4D rotations using rotors
    static func testRotations() {
        print("\nTesting 4D rotations...")

        // Create a 4D point
        let point = SIMD4<Float>(1.0, 0.0, 0.0, 0.0)
        print("Original point: \(point)")

        // Create rotation in XY plane (90 degrees)
        let rotorXY = GA4D.Metric.Operations4D.rotationXY(angle: Float.pi / 2)
        print("Rotor for 90° XY rotation: \(rotorXY)")

        // Apply the rotation
        let rotatedPoint = GA4D.Metric.Operations4D.rotate(vector: point, using: rotorXY)
        print("Point after XY rotation: \(rotatedPoint)")

        // Create rotation in YZ plane (45 degrees)
        let rotorYZ = GA4D.Metric.Operations4D.rotationYZ(angle: Float.pi / 4)
        print("Rotor for 45° YZ rotation: \(rotorYZ)")

        // Apply both rotations in sequence
        let combinedRotated = GA4D.Metric.Operations4D.applyRotorSequence(
            to: point,
            rotors: [rotorXY, rotorYZ]
        )
        print("Point after combined rotations: \(combinedRotated)")

        // Create a rotation that moves one vector to another
        let source = SIMD4<Float>(1.0, 0.0, 0.0, 0.0)
        let target = SIMD4<Float>(0.0, 0.0, 1.0, 0.0)

        print("\nRotating from \(source) to \(target)...")

        // Create a rotor to rotate source to target
        let customRotor = GA4D.Metric.Operations4D.rotorBetweenVectors(from: source, to: target)
        print("Custom rotor: \(customRotor)")

        // Apply the custom rotation
        let customRotated = GA4D.Metric.Operations4D.rotate(vector: source, using: customRotor)
        print("Result of custom rotation: \(customRotated)")

        // Should be approximately equal to target
        let isCloseToTarget = simd_distance(customRotated, target) < 1e-5
        print("Is result close to target? \(isCloseToTarget)")
    }

    // MARK: - 4D Reflections

    /// Test 4D reflections using versors
    static func testReflections() {
        print("\nTesting 4D reflections...")

        // Create a 4D point
        let point = SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
        print("Original point: \(point)")

        // Create a normal to a hyperplane
        let normal = normalize(SIMD4<Float>(1.0, 0.0, 0.0, 0.0))
        print("Hyperplane normal: \(normal)")

        // Reflect the point in the hyperplane
        let reflected = GA4D.Metric.Operations4D.reflect(vector: point, in: normal)
        print("Reflected point: \(reflected)")

        // Create reflection in a different hyperplane
        let normal2 = normalize(SIMD4<Float>(1.0, 1.0, 0.0, 0.0))
        print("Second hyperplane normal: \(normal2)")

        // Reflect in the second hyperplane
        let reflected2 = GA4D.Metric.Operations4D.reflect(vector: point, in: normal2)
        print("Point reflected in second hyperplane: \(reflected2)")
    }

    // MARK: - 4D Geometric Primitives

    /// Test 4D geometric primitives
    static func testPrimitives() {
        print("\nTesting 4D geometric primitives...")

        // Create a tesseract (using 24-cell as an approximation)
        let tesseractVertices = GA4D.AdvancedPrimitives.create24Cell(scale: 1.0)
        print("Tesseract has \(tesseractVertices.count) vertices")

        // Create edges for the tesseract
        let tesseractEdges = GA4D.AdvancedPrimitives.create24CellEdges(vertices: tesseractVertices)
        print("Tesseract has \(tesseractEdges.count) edges")

        // Create a hypersphere (using Hopf fibration)
        let hypersphereVertices = GA4D.AdvancedPrimitives.createHopfFibration(
            radius: 1.0, fiberCount: 16, pointsPerFiber: 16)
        print("Hypersphere has \(hypersphereVertices.count) vertices")

        // Create a Clifford torus
        let torusVertices = GA4D.AdvancedPrimitives.create4DTorus(
            majorRadius: 1.0,
            minorRadius: 0.5,
            resolution: 16
        )
        print("Clifford torus has \(torusVertices.count) vertices")
    }

    // MARK: - 4D to 3D Projection

    /// Test projections from 4D to 3D
    static func testProjections() {
        print("\nTesting 4D to 3D projections...")

        // Create a simple 4D cube
        let cubeVertices = GA4D.AdvancedPrimitives.create24Cell(scale: 1.0)
        print("Created tesseract with \(cubeVertices.count) vertices")

        // Create rotation angles for each plane
        let rotations: (xy: Float, xz: Float, xw: Float, yz: Float, yw: Float, zw: Float) = (
            xy: Float.pi / 6,
            xz: Float.pi / 4,
            xw: Float.pi / 3,
            yz: 0.0,
            yw: Float.pi / 8,
            zw: 0.0
        )

        // Project using stereographic projection
        let stereographicVertices = GA4D.GA4DMetalBridge.transformVertices(
            vertices4D: cubeVertices,
            rotations: rotations,
            projectionType: .stereographic
        )
        print("Stereographic projection created \(stereographicVertices.count) 3D vertices")

        // Project using perspective projection
        let perspectiveVertices = GA4D.GA4DMetalBridge.transformVertices(
            vertices4D: cubeVertices,
            rotations: rotations,
            projectionType: .perspective
        )
        print("Perspective projection created \(perspectiveVertices.count) 3D vertices")

        // Project using orthographic projection
        let orthographicVertices = GA4D.GA4DMetalBridge.transformVertices(
            vertices4D: cubeVertices,
            rotations: rotations,
            projectionType: .orthographic
        )
        print("Orthographic projection created \(orthographicVertices.count) 3D vertices")

        // Compare the first vertex in each projection
        if stereographicVertices.count > 0 && perspectiveVertices.count > 0
            && orthographicVertices.count > 0
        {
            print("\nProjection comparison for first vertex:")
            print("Stereographic: \(stereographicVertices[0])")
            print("Perspective: \(perspectiveVertices[0])")
            print("Orthographic: \(orthographicVertices[0])")
        }
    }

    // MARK: - Normal Calculation

    /// Test 4D normal calculation using GA
    static func testNormalCalculation() {
        print("\nTesting 4D normal calculation...")

        // Create a simple tetrahedron in 4D
        let vertices4D: [SIMD4<Float>] = [
            SIMD4<Float>(0.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(1.0, 0.0, 0.0, 0.0),
            SIMD4<Float>(0.0, 1.0, 0.0, 0.0),
            SIMD4<Float>(0.0, 0.0, 1.0, 0.0),
        ]
        print("Created tetrahedron with \(vertices4D.count) vertices")

        // Define triangular faces
        let triangles: [(Int, Int, Int)] = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 1),
            (1, 3, 2),
        ]
        print("Created \(triangles.count) triangular faces")

        // Calculate normals
        let normals = GA4D.GA4DMetalBridge.calculateNormals(for: vertices4D, connections: triangles)
        print("Calculated \(normals.count) normals")

        // Print the normals
        for i in 0..<min(4, normals.count) {
            print("Normal \(i): \(normals[i]), length: \(simd_length(normals[i]))")
        }
    }

    // MARK: - Translation and General Transformations

    /// Test translations in 4D using GA
    static func testTranslations() {
        print("\nTesting 4D translations...")

        // Create a 4D point
        let point = SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
        print("Original point: \(point)")

        // Create a translator in the (1,1,1) direction with distance 2
        let translator = GA4D.Metric.Operations4D.translator(
            direction: SIMD3<Float>(1.0, 1.0, 1.0),
            distance: 2.0
        )
        print("Created translator: \(translator)")

        // Apply the translation
        let translatedPoint = translator.transform(GA4D.Metric.Multivector.vector(point))
            .toVector()!
        print("Translated point: \(translatedPoint)")

        // Verify the translation was in the right direction
        let expectedPoint = SIMD4<Float>(3.0, 3.0, 3.0, 1.0)
        let closeEnough = simd_distance(translatedPoint, expectedPoint) < 0.1
        print("Is result close to expected? \(closeEnough)")
    }

    // MARK: - Complete Integration Example

    /// Comprehensive example that integrates all features
    static func runCompleteExample() {
        print("\n=== Running complete GA4D integration example ===")

        // Create a 4D shape (hypersphere)
        let vertices4D = GA4D.AdvancedPrimitives.createHopfFibration(
            radius: 1.0, fiberCount: 16, pointsPerFiber: 16)
        print("Created hypersphere with \(vertices4D.count) vertices")

        // Create rotors for animation
        let timeStep: Float = 0.1
        var rotationAngles: (xy: Float, xz: Float, xw: Float, yz: Float, yw: Float, zw: Float) = (
            0, 0, 0, 0, 0, 0
        )

        // Simulate an animation sequence
        for frame in 0..<5 {
            // Update rotation angles
            rotationAngles.xy += 0.1 * timeStep
            rotationAngles.xw += 0.2 * timeStep
            rotationAngles.yw += 0.05 * timeStep

            print("\nFrame \(frame):")
            print(
                "Rotation angles: XY=\(rotationAngles.xy), XW=\(rotationAngles.xw), YW=\(rotationAngles.yw)"
            )

            // Project to 3D
            let vertices3D = GA4D.GA4DMetalBridge.transformVertices(
                vertices4D: vertices4D,
                rotations: rotationAngles,
                projectionType: .stereographic
            )

            // Calculate surface properties
            let triangleCount = min(10, vertices3D.count / 3)
            var triangles: [(Int, Int, Int)] = []

            // Create sample triangulation
            for i in 0..<triangleCount {
                triangles.append((i * 3, i * 3 + 1, i * 3 + 2))
            }

            // Calculate normals
            let normals4D = GA4D.GA4DMetalBridge.calculateNormals(
                for: vertices4D, connections: triangles)

            // Report statistics
            print("Projected to \(vertices3D.count) 3D vertices")
            print("Calculated \(normals4D.count) normals")

            // Calculate bounding box
            if !vertices3D.isEmpty {
                var minBounds = vertices3D[0]
                var maxBounds = vertices3D[0]

                for vertex in vertices3D {
                    minBounds = min(minBounds, vertex)
                    maxBounds = max(maxBounds, vertex)
                }

                print("3D bounding box: Min=\(minBounds), Max=\(maxBounds)")
            }
        }
    }

    // MARK: - Performance Testing

    /// Performance test for GA operations
    static func runPerformanceTests() {
        print("\n=== Running GA4D performance tests ===")

        // Test creation of multivectors
        let startCreation = Date()
        var multivectors: [GA4D.Metric.Multivector] = []

        for i in 0..<1000 {
            let scalar = Float(i % 10) / 10.0
            let vector = SIMD4<Float>(
                Float(i % 5) / 5.0,
                Float((i + 1) % 7) / 7.0,
                Float((i + 2) % 6) / 6.0,
                Float((i + 3) % 8) / 8.0
            )

            multivectors.append(
                GA4D.Metric.Multivector.scalar(scalar, dimension: GA4D.Dimension.dim4))
            multivectors.append(GA4D.Metric.Multivector.vector(vector))
        }

        let creationTime = Date().timeIntervalSince(startCreation)
        print("Created 2000 multivectors in \(creationTime) seconds")

        // Test geometric product
        let startProduct = Date()
        var products: [GA4D.Metric.Multivector] = []

        for i in 0..<(multivectors.count - 1) {
            products.append(multivectors[i] * multivectors[i + 1])
        }

        let productTime = Date().timeIntervalSince(startProduct)
        print("Computed \(products.count) geometric products in \(productTime) seconds")

        // Test 4D rotations
        let startRotations = Date()
        let testVector = SIMD4<Float>(1.0, 0.0, 0.0, 0.0)
        var rotatedVectors: [SIMD4<Float>] = []

        for i in 0..<360 {
            let angle = Float(i) * Float.pi / 180.0
            let rotorXY = GA4D.Metric.Operations4D.rotationXY(angle: angle)
            let rotorXW = GA4D.Metric.Operations4D.rotationXW(angle: angle * 0.5)

            let rotated = GA4D.Metric.Operations4D.applyRotorSequence(
                to: testVector,
                rotors: [rotorXY, rotorXW]
            )

            rotatedVectors.append(rotated)
        }

        let rotationTime = Date().timeIntervalSince(startRotations)
        print("Computed \(rotatedVectors.count) 4D rotations in \(rotationTime) seconds")

        // Test 4D to 3D projection
        let startProjection = Date()
        let vertices4D = GA4D.AdvancedPrimitives.createHopfFibration(
            radius: 1.0, fiberCount: 32, pointsPerFiber: 32)

        let rotations: (xy: Float, xz: Float, xw: Float, yz: Float, yw: Float, zw: Float) = (
            Float.pi / 4, Float.pi / 6, Float.pi / 3, Float.pi / 8, Float.pi / 5, Float.pi / 7
        )

        // Store the result in a variable we'll use
        let vertices3D = GA4D.GA4DMetalBridge.transformVertices(
            vertices4D: vertices4D,
            rotations: rotations,
            projectionType: .stereographic
        )

        let projectionTime = Date().timeIntervalSince(startProjection)
        print("Projected \(vertices4D.count) 4D vertices to 3D in \(projectionTime) seconds")
        print("Resulting in \(vertices3D.count) 3D vertices")

        // Overall performance assessment
        print("\nPerformance summary:")
        print("- Multivector creation: \(creationTime * 1000 / 2000) ms per multivector")
        print("- Geometric product: \(productTime * 1000 / Double(products.count)) ms per product")
        print("- 4D rotation: \(rotationTime * 1000 / 360) ms per rotation")
        print(
            "- 4D to 3D projection: \(projectionTime * 1000 / Double(vertices4D.count)) ms per vertex"
        )
    }

    // MARK: - Run All Tests

    /// Run all test methods
    static func runAllTests() {
        print("=== GA4D TEST SUITE ===")
        testBasicOperations()
        testRotations()
        testReflections()
        testPrimitives()
        testProjections()
        testNormalCalculation()
        testTranslations()
        runCompleteExample()
        runPerformanceTests()
        print("=== END OF TESTS ===")
    }
}
