//
//  GA4DOptimizer.swift
//  Graf
//
//  Created on 15/03/2025
//  Performance optimization for GA4D operations
//

import Accelerate
import Foundation
import Metal
import simd

/// Performance optimization for Geometric Algebra operations
public class GA4DOptimizer {

    // MARK: - Types

    /// Performance options for GA operations
    public struct PerformanceOptions: OptionSet {
        public let rawValue: Int

        public init(rawValue: Int) {
            self.rawValue = rawValue
        }

        /// Use Metal for parallel processing of GA operations
        public static let useMetal = PerformanceOptions(rawValue: 1 << 0)

        /// Use Accelerate framework for vector operations
        public static let useAccelerate = PerformanceOptions(rawValue: 1 << 1)

        /// Cache product tables for repeated operations
        public static let cacheProductTables = PerformanceOptions(rawValue: 1 << 2)

        /// Use optimized algorithms for rotors
        public static let optimizeRotors = PerformanceOptions(rawValue: 1 << 3)

        /// Batch multiple operations together
        public static let batchOperations = PerformanceOptions(rawValue: 1 << 4)

        /// All optimizations enabled
        public static let all: PerformanceOptions = [
            .useMetal, .useAccelerate, .cacheProductTables, .optimizeRotors, .batchOperations,
        ]
    }

    /// Statistics about GA operations for performance tuning
    public struct PerformanceStatistics {
        /// Total number of geometric products computed
        public var geometricProductCount: Int = 0

        /// Total number of wedge products computed
        public var wedgeProductCount: Int = 0

        /// Total number of inner products computed
        public var innerProductCount: Int = 0

        /// Total time spent on geometric products (seconds)
        public var geometricProductTime: Double = 0

        /// Total time spent on wedge products (seconds)
        public var wedgeProductTime: Double = 0

        /// Total time spent on inner products (seconds)
        public var innerProductTime: Double = 0

        /// Total number of rotor applications
        public var rotorApplicationCount: Int = 0

        /// Total time spent on rotor applications (seconds)
        public var rotorApplicationTime: Double = 0

        /// Reset all statistics to zero
        public mutating func reset() {
            geometricProductCount = 0
            wedgeProductCount = 0
            innerProductCount = 0
            geometricProductTime = 0
            wedgeProductTime = 0
            innerProductTime = 0
            rotorApplicationCount = 0
            rotorApplicationTime = 0
        }
    }

    // MARK: - Properties

    /// The Metal device to use for computations
    private var device: MTLDevice?

    /// The command queue for Metal computations
    private var commandQueue: MTLCommandQueue?

    /// The compute pipeline state for GA operations
    private var computePipelineState: MTLComputePipelineState?

    /// Current performance options
    private var options: PerformanceOptions

    /// Performance statistics
    private(set) public var statistics = PerformanceStatistics()

    /// Cached rotors for frequent operations
    private var cachedRotors: [String: GA4D.Metric.Multivector] = [:]

    /// Cached product tables for different dimensions and metrics
    private var cachedProductTables: [String: Any] = [:]

    // MARK: - Initialization

    /// Initialize the optimizer with performance options
    /// - Parameter options: The performance options to use
    public init(options: PerformanceOptions = .all) {
        self.options = options

        // Initialize Metal if requested
        if options.contains(.useMetal) {
            self.device = MTLCreateSystemDefaultDevice()
            self.commandQueue = device?.makeCommandQueue()
            setupMetal()
        } else {
            self.device = nil
            self.commandQueue = nil
        }
    }

    /// Set up Metal for GA computations
    private func setupMetal() {
        guard let device = device else { return }

        // Load GA compute shader from default library
        guard let library = device.makeDefaultLibrary() else {
            print("Error: Could not load default Metal library")
            return
        }

        // Look for the GA compute function
        guard let computeFunction = library.makeFunction(name: "geometricProductCompute") else {
            print("Error: Could not find geometricProductCompute function in Metal library")
            return
        }

        // Create compute pipeline state
        do {
            computePipelineState = try device.makeComputePipelineState(function: computeFunction)
        } catch {
            print("Error: Could not create compute pipeline state: \(error)")
        }
    }

    // MARK: - Optimization Methods

    /// Optimize an array of rotors for efficient application
    /// - Parameter rotors: Array of rotors to optimize
    /// - Returns: Optimized rotors array
    public func optimizeRotors(_ rotors: [GA4D.Metric.Multivector]) -> [GA4D.Metric.Multivector] {
        guard options.contains(.optimizeRotors) else {
            return rotors
        }

        // If we only have one rotor, just normalize it
        if rotors.count == 1 {
            return [rotors[0].normalized()]
        }

        // Combine rotors if possible
        var combinedRotor = rotors[0]
        for i in 1..<rotors.count {
            combinedRotor = GA4D.Metric.GeometricProduct.geometricProduct(combinedRotor, rotors[i])
        }

        // Normalize combined rotor
        return [combinedRotor.normalized()]
    }

    /// Cache a rotor for frequent use
    /// - Parameters:
    ///   - rotor: The rotor to cache
    ///   - key: A key to identify the rotor
    public func cacheRotor(_ rotor: GA4D.Metric.Multivector, forKey key: String) {
        cachedRotors[key] = rotor
    }

    /// Get a cached rotor by key
    /// - Parameter key: The key identifying the rotor
    /// - Returns: The cached rotor if available
    public func getCachedRotor(forKey key: String) -> GA4D.Metric.Multivector? {
        return cachedRotors[key]
    }

    // MARK: - Batch Operations

    /// Apply a rotor to multiple vertices in batch
    /// - Parameters:
    ///   - vertices: Array of 4D vertices
    ///   - rotor: The rotor to apply
    /// - Returns: Array of transformed vertices
    public func batchApplyRotor(to vertices: [SIMD4<Float>], rotor: GA4D.Metric.Multivector)
        -> [SIMD4<
            Float
        >]
    {
        let startTime = CFAbsoluteTimeGetCurrent()
        defer {
            statistics.rotorApplicationTime += CFAbsoluteTimeGetCurrent() - startTime
            statistics.rotorApplicationCount += vertices.count
        }

        // Use Metal for batch processing if available
        if options.contains(.useMetal) && device != nil && computePipelineState != nil {
            return batchApplyRotorMetal(to: vertices, rotor: rotor)
        }

        // Use Accelerate for batch processing if available
        if options.contains(.useAccelerate) {
            return batchApplyRotorAccelerate(to: vertices, rotor: rotor)
        }

        // Fall back to standard processing
        var result = [SIMD4<Float>]()
        result.reserveCapacity(vertices.count)

        for vertex in vertices {
            result.append(GA4D.Metric.Operations4D.rotate(vector: vertex, using: rotor))
        }

        return result
    }

    /// Apply a rotor to multiple vertices using Metal
    private func batchApplyRotorMetal(to vertices: [SIMD4<Float>], rotor: GA4D.Metric.Multivector)
        -> [SIMD4<Float>]
    {
        guard let device = device,
            let commandQueue = commandQueue,
            let computePipelineState = computePipelineState
        else {
            // Fall back to CPU if Metal setup is incomplete
            return vertices.map { GA4D.Metric.Operations4D.rotate(vector: $0, using: rotor) }
        }

        // Create buffer for vertices
        guard
            let vertexBuffer = device.makeBuffer(
                bytes: vertices,
                length: vertices.count * MemoryLayout<SIMD4<Float>>.stride,
                options: .storageModeShared
            )
        else {
            return vertices.map { GA4D.Metric.Operations4D.rotate(vector: $0, using: rotor) }
        }

        // Create buffer for output vertices
        guard
            let outputBuffer = device.makeBuffer(
                length: vertices.count * MemoryLayout<SIMD4<Float>>.stride,
                options: .storageModeShared
            )
        else {
            return vertices.map { GA4D.Metric.Operations4D.rotate(vector: $0, using: rotor) }
        }

        // Create buffer for rotor components
        let rotorComponents = rotor.allComponents
        guard
            let rotorBuffer = device.makeBuffer(
                bytes: rotorComponents,
                length: rotorComponents.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        else {
            return vertices.map { GA4D.Metric.Operations4D.rotate(vector: $0, using: rotor) }
        }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return vertices.map { GA4D.Metric.Operations4D.rotate(vector: $0, using: rotor) }
        }

        // Create compute command encoder
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return vertices.map { GA4D.Metric.Operations4D.rotate(vector: $0, using: rotor) }
        }

        // Set compute pipeline state
        computeEncoder.setComputePipelineState(computePipelineState)

        // Set buffers
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(rotorBuffer, offset: 0, index: 2)

        // Calculate threads and threadgroups
        let gridSize = MTLSize(width: vertices.count, height: 1, depth: 1)
        let threadGroupSize = MTLSize(
            width: min(computePipelineState.maxTotalThreadsPerThreadgroup, vertices.count),
            height: 1,
            depth: 1
        )

        // Dispatch threads
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

        // End encoding
        computeEncoder.endEncoding()

        // Commit command buffer
        commandBuffer.commit()

        // Wait for completion
        commandBuffer.waitUntilCompleted()

        // Read results back
        let outputPtr = outputBuffer.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: vertices.count)
        var result = [SIMD4<Float>]()
        result.reserveCapacity(vertices.count)

        for i in 0..<vertices.count {
            result.append(outputPtr[i])
        }

        return result
    }

    /// Apply a rotor to multiple vertices using Accelerate
    private func batchApplyRotorAccelerate(
        to vertices: [SIMD4<Float>], rotor: GA4D.Metric.Multivector
    )
        -> [SIMD4<Float>]
    {
        // Extract rotor components
        // For this optimized implementation, we'll use the fact that a rotor R = a + B
        // where a is the scalar part and B is the bivector part
        // The sandwich product R*v*R̃ can be implemented more efficiently

        // Extract bivector components (we're assuming a 4D rotor)
        var bivectorParts = [Float](repeating: 0, count: 6)  // 6 bivector components in 4D

        // Map bivector indices to array positions
        let bivectorIndices = [
            GA4D.Metric.Multivector.indexFromBasisString("e12", dimension: .dim4),
            GA4D.Metric.Multivector.indexFromBasisString("e13", dimension: .dim4),
            GA4D.Metric.Multivector.indexFromBasisString("e14", dimension: .dim4),
            GA4D.Metric.Multivector.indexFromBasisString("e23", dimension: .dim4),
            GA4D.Metric.Multivector.indexFromBasisString("e24", dimension: .dim4),
            GA4D.Metric.Multivector.indexFromBasisString("e34", dimension: .dim4),
        ]

        for (i, idx) in bivectorIndices.enumerated() {
            bivectorParts[i] = rotor[idx]
        }

        // Prepare result array
        var result = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: vertices.count)

        // Flatten input vertices for Accelerate
        var flattenedInput = [Float](repeating: 0, count: vertices.count * 4)
        for i in 0..<vertices.count {
            flattenedInput[i * 4] = vertices[i].x
            flattenedInput[i * 4 + 1] = vertices[i].y
            flattenedInput[i * 4 + 2] = vertices[i].z
            flattenedInput[i * 4 + 3] = vertices[i].w
        }

        // Prepare output buffer
        let _ = [Float](repeating: 0, count: vertices.count * 4)

        // Apply optimized rotor transformation
        // This is a simplified implementation that works for common cases
        // A full implementation would handle all possible bivector combinations

        // Apply transformation using Accelerate framework
        // This is a placeholder for the actual implementation
        // In a real implementation, we'd write efficient BLAS operations

        // For now, just fall back to regular computation
        for i in 0..<vertices.count {
            result[i] = GA4D.Metric.Operations4D.rotate(vector: vertices[i], using: rotor)
        }

        return result
    }

    // MARK: - Metal Compute Functions

    /// Generate Metal shader code for geometric algebra operations
    public func generateMetalShaderCode() -> String {
        // This is a template for the Metal compute kernel that performs
        // geometric product and rotor application

        return """
            #include <metal_stdlib>
            using namespace metal;

            // Structure for a 4D vector
            struct Vector4 {
                float x, y, z, w;
            };

            // Structure for a multivector in 4D (16 components)
            struct Multivector {
                float components[16];
            };

            // Apply a rotor to a 4D vector
            kernel void geometricProductCompute(
                device const Vector4* vertices [[buffer(0)]],
                device Vector4* result [[buffer(1)]],
                device const float* rotorComponents [[buffer(2)]],
                uint id [[thread_position_in_grid]]
            ) {
                // Get the input vertex
                Vector4 vertex = vertices[id];
                
                // Create vector multivector (grade 1)
                Multivector vMultivector;
                for (int i = 0; i < 16; i++) {
                    vMultivector.components[i] = 0.0;
                }
                vMultivector.components[1] = vertex.x;  // e1
                vMultivector.components[2] = vertex.y;  // e2
                vMultivector.components[4] = vertex.z;  // e3
                vMultivector.components[8] = vertex.w;  // e4
                
                // Create rotor multivector (grade 0 + grade 2)
                Multivector rotorMultivector;
                for (int i = 0; i < 16; i++) {
                    rotorMultivector.components[i] = rotorComponents[i];
                }
                
                // Compute the sandwich product R * v * R̃
                // This is a simplified implementation for demonstration
                
                // For efficiency, we'll directly compute the vector output
                // This is equivalent to implementing the full geometric product
                // but optimized for the specific case of vector rotation
                
                // For now, implement a simplified version that assumes
                // the rotor is only in the XY plane for demonstration
                
                float scalar = rotorMultivector.components[0];  // Scalar part
                float xy = rotorMultivector.components[3];      // e12 component
                
                // Apply a simple rotation in the XY plane
                float x = vertex.x;
                float y = vertex.y;
                
                float cosTheta = scalar;
                float sinTheta = xy;
                
                Vector4 rotated;
                rotated.x = x * cosTheta - y * sinTheta;
                rotated.y = x * sinTheta + y * cosTheta;
                rotated.z = vertex.z;
                rotated.w = vertex.w;
                
                // Store the result
                result[id] = rotated;
            }
            """
    }

    // MARK: - Performance Monitoring

    /// Reset performance statistics
    public func resetStatistics() {
        statistics.reset()
    }

    /// Get performance recommendations based on statistics
    /// - Returns: Array of recommendation strings
    public func getPerformanceRecommendations() -> [String] {
        var recommendations: [String] = []

        // Check for frequent geometric products
        if statistics.geometricProductCount > 10000 {
            recommendations.append(
                "High geometric product count (\(statistics.geometricProductCount)). Consider caching results or using batch operations."
            )
        }

        // Check for slow geometric products
        let averageGPTime =
            statistics.geometricProductCount > 0
            ? statistics.geometricProductTime / Double(statistics.geometricProductCount) : 0
        if averageGPTime > 0.0001 {  // 100 microseconds threshold
            recommendations.append(
                "Slow geometric products (avg \(String(format: "%.2f", averageGPTime * 1_000_000)) μs). Consider using Metal acceleration."
            )
        }

        // Check for frequent rotor applications
        if statistics.rotorApplicationCount > 1000 {
            recommendations.append(
                "High rotor application count (\(statistics.rotorApplicationCount)). Consider combining rotors or using batch transformations."
            )
        }

        // Check for Metal availability
        if options.contains(.useMetal) && device == nil {
            recommendations.append(
                "Metal acceleration requested but not available. Consider falling back to Accelerate framework."
            )
        }

        // If no specific recommendations, provide a general note
        if recommendations.isEmpty {
            recommendations.append("Performance is optimal based on current statistics.")
        }

        return recommendations
    }

    /// Set performance options
    /// - Parameter options: The new performance options
    public func setOptions(_ options: PerformanceOptions) {
        self.options = options

        // Update Metal setup if needed
        if options.contains(.useMetal) && device == nil {
            // Try to initialize Metal
            let newDevice = MTLCreateSystemDefaultDevice()
            if newDevice != nil {
                self.device = newDevice
                self.commandQueue = newDevice?.makeCommandQueue()
                setupMetal()
            }
        }
    }
}

// MARK: - Swizzling Extensions

extension SIMD4 where Scalar == Float {
    /// Extract specified components to form a SIMD3
    /// - Parameter indices: The indices to extract (0=x, 1=y, 2=z, 3=w)
    /// - Returns: A SIMD3 with the specified components
    func swizzle(_ indices: (Int, Int, Int)) -> SIMD3<Float> {
        let components = [self[0], self[1], self[2], self[3]]
        return SIMD3<Float>(
            components[indices.0],
            components[indices.1],
            components[indices.2]
        )
    }
}
