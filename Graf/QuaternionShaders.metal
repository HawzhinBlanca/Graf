//
//  VertexIn.swift
//  Graf
//
//  Created by HAWZHIN on 04/03/2025.
//


#include <metal_stdlib>
#include <metal_geometric>
using namespace metal;

// Vertex input structure
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float4 color [[attribute(2)]];
};

// Vertex output structure
struct VertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 normal;
    float4 color;
    float3 viewDirection;
};

// Uniform structure
struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float3x3 normalMatrix;
    float4 color;
    float4 options;
};

// Line shader for wireframe
vertex VertexOut lineVertexShader(const VertexIn in [[stage_in]],
                                 constant Uniforms &uniforms [[buffer(1)]]) {
    // Position in world space
    float4 worldPosition = uniforms.modelMatrix * float4(in.position, 1.0);
    
    // Position in clip space
    float4 clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;
    
    // Output vertex data
    VertexOut out;
    out.position = clipPosition;
    out.worldPosition = worldPosition.xyz;
    out.normal = normalize(uniforms.normalMatrix * in.normal);
    out.color = uniforms.color; // Use the uniform color for lines
    out.viewDirection = float3(0.0, 0.0, 1.0); // Default view direction
    
    return out;
}

// Fragment shader for lines
fragment float4 lineFragmentShader(VertexOut in [[stage_in]],
                                  constant Uniforms &uniforms [[buffer(1)]]) {
    // Simple shader that just outputs the color
    return in.color;
}

// Point shader
vertex VertexOut pointVertexShader(const VertexIn in [[stage_in]],
                                  constant Uniforms &uniforms [[buffer(1)]]) {
    // Position in world space
    float4 worldPosition = uniforms.modelMatrix * float4(in.position, 1.0);
    
    // Position in clip space
    float4 clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;
    
    // Output vertex data
    VertexOut out;
    out.position = clipPosition;
    out.worldPosition = worldPosition.xyz;
    out.normal = normalize(uniforms.normalMatrix * in.normal);
    out.color = uniforms.color; // Use uniform color for points
    out.viewDirection = float3(0.0, 0.0, 1.0); // Default view direction
    
    return out;
}

// Fragment shader for points
fragment float4 pointFragmentShader(VertexOut in [[stage_in]],
                                   constant Uniforms &uniforms [[buffer(1)]]) {
    // Add a glow effect for points
    float glow = 1.2;
    float3 glowColor = in.color.rgb * glow;
    
    return float4(glowColor, in.color.a);
}
