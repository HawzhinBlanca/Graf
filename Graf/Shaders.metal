#include <metal_stdlib>
#include <metal_geometric>
#include <metal_math>
using namespace metal;

// Constants for shader configuration
/*
constant float kAmbientIntensity = 0.3;
constant float kDiffuseIntensity = 0.7;
constant float kSpecularIntensity = 0.5;
constant float kSpecularPower = 64.0;
constant float kEdgeIntensity = 0.5;
*/
constant float kRimPower = 3.0;
constant float kRimIntensity = 0.4;

// PBR constants
constant float kMinRoughness = 0.04;
constant float PI = 3.1415926535897932384626433832795;

// Structures
struct Vertex3D {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float4 color [[attribute(2)]];
    float2 texCoord [[attribute(3)]];
};

struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float3x3 normalMatrix;
    float time;
    uint4 options;  // Packed options as integers
                    // x: wireframe mode (0/1)
                    // y: show normals (0/1)
                    // z: special effects (0/1)
                    // w: extra visualization mode (0-n)
};

struct LightingParams {
    float3 lightPosition;
    float3 lightColor;
    float lightIntensity;
    float3 ambientColor;
    float ambientIntensity;
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 normal;
    float4 color;
    float2 texCoord;
    float3 viewDirection;
    float3 lightDirection;
    float4 projPosition; // For depth calculations
    float wComponent;    // Original 4D w-component
};

// PBR Material Definition
struct PBRMaterial {
    float3 albedo;
    float metallic;
    float roughness;
    float ambientOcclusion;
    float3 emissive;
};

// Utility Functions
float3 getRainbowColor(float value) {
    // Map value (0-1) to rainbow color
    float hue = fract(value) * 6.0;  // Cycle through hues (0-6)
    float X = 1.0 - abs(fmod(hue, 2.0) - 1.0);
    
    if (hue < 1.0) return float3(1.0, X, 0.0);
    else if (hue < 2.0) return float3(X, 1.0, 0.0);
    else if (hue < 3.0) return float3(0.0, 1.0, X);
    else if (hue < 4.0) return float3(0.0, X, 1.0);
    else if (hue < 5.0) return float3(X, 0.0, 1.0);
    else return float3(1.0, 0.0, X);
}

// Function for hyperspatial coloring based on 4D coordinates
float4 hyperspatialColor(float wCoord, float time) {
    // Create interesting patterns based on 4D coordinates
    float3 baseColor = getRainbowColor(wCoord * 0.5 + time * 0.1);
    float pulse = sin(time * 0.5 + wCoord * 3.0) * 0.5 + 0.5;
    
    return float4(baseColor * pulse, 1.0);
}

// PBR Functions
// Normal Distribution Function (GGX/Trowbridge-Reitz)
float distributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return nom / max(denom, 0.0000001);
}

// Geometry Function (Smith's method with GGX)
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return nom / max(denom, 0.0000001);
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// Fresnel Function (Schlick's approximation)
float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Vertex shader
vertex VertexOut vertexShader(const Vertex3D in [[stage_in]],
                             constant Uniforms &uniforms [[buffer(1)]]) {
    // Extract camera position from view matrix (inverse of camera transform)
    float3 cameraPosition = float3(
        -(uniforms.viewMatrix[0].w * uniforms.viewMatrix[0].x + 
          uniforms.viewMatrix[1].w * uniforms.viewMatrix[1].x + 
          uniforms.viewMatrix[2].w * uniforms.viewMatrix[2].x),
        -(uniforms.viewMatrix[0].w * uniforms.viewMatrix[0].y + 
          uniforms.viewMatrix[1].w * uniforms.viewMatrix[1].y + 
          uniforms.viewMatrix[2].w * uniforms.viewMatrix[2].y),
        -(uniforms.viewMatrix[0].w * uniforms.viewMatrix[0].z + 
          uniforms.viewMatrix[1].w * uniforms.viewMatrix[1].z + 
          uniforms.viewMatrix[2].w * uniforms.viewMatrix[2].z)
    );
    
    // Store the original w component for 4D effects
    float wComponent = in.position.z;  // Assuming w is mapped to z for visualization
    
    // Apply model transformation
    float4 worldPosition = uniforms.modelMatrix * float4(in.position, 1.0);
    
    // Apply animation based on visualization mode
    float animationFactor = 1.0;
    float angle = 0.0; // Declare angle variable outside switch statement
    
    switch(uniforms.options.w) {  // Extra visualization mode
        case 1:  // Pulsating effect
            animationFactor = sin(uniforms.time * 0.5) * 0.05 + 1.0;
            break;
        case 2:  // Wave effect
            animationFactor = 1.0 + sin(worldPosition.x * 2.0 + uniforms.time) * 0.1;
            break;
        case 3:  // Spiral effect
            angle = atan2(worldPosition.y, worldPosition.x) + uniforms.time * 0.5;
            animationFactor = 1.0 + sin(angle * 5.0) * 0.05;
            break;
        default:  // Default subtle breathing
            animationFactor = sin(uniforms.time * 0.2) * 0.02 + 1.0;
    }
    
    // Apply the animation factor
    worldPosition.xyz *= animationFactor;
    
    // Transform to clip space
    float4 viewPosition = uniforms.viewMatrix * worldPosition;
    float4 clipPosition = uniforms.projectionMatrix * viewPosition;
    
    // Transform normal to world space
    float3 worldNormal = normalize(uniforms.normalMatrix * in.normal);
    
    // Handle normal flipping for negative scaling
    if (determinant(uniforms.modelMatrix) < 0.0) {
        worldNormal = -worldNormal;
    }
    
    // Calculate view and light directions in world space
    float3 lightPosition = float3(20.0, 20.0, 20.0);  // Main light source
    float3 lightDirection = normalize(lightPosition - worldPosition.xyz);
    float3 viewDirection = normalize(cameraPosition - worldPosition.xyz);
    
    // Prepare vertex output
    VertexOut out;
    out.position = clipPosition;
    out.projPosition = clipPosition; // Store for depth calculations
    out.worldPosition = worldPosition.xyz;
    out.normal = worldNormal;
    out.viewDirection = viewDirection;
    out.lightDirection = lightDirection;
    out.wComponent = wComponent;  // Pass the w component for 4D effects
    
    // Apply color effects based on mode
    float4 color = in.color;
    
    if (uniforms.options.x == 0) {  // Non-wireframe mode
        // Enhance colors based on time
        float colorPulse = sin(uniforms.time * 0.2) * 0.2 + 0.8;
        color *= colorPulse;
        
        // Add 4D coordinate influence
        if (uniforms.options.z > 0) {  // If special effects enabled
            float4 hyperColor = hyperspatialColor(out.wComponent, uniforms.time);
            color = mix(color, hyperColor, 0.3);
        }
    } else {
        // Wireframe mode - use distance from origin for color variation
        float distFactor = length(in.position) * 0.2;
        float3 wireColor = getRainbowColor(distFactor + uniforms.time * 0.1);
        color = mix(color, float4(wireColor, 1.0), 0.4);
    }
    
    out.color = color;
    out.texCoord = in.texCoord;
    
    return out;
}

// Fragment shader
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                              constant Uniforms &uniforms [[buffer(1)]]) {
    
    // Calculate derivatives for edge detection and normal mapping
    float3 dpdx = dfdx(in.worldPosition);
    float3 dpdy = dfdy(in.worldPosition);
    float edgeStrength = length(cross(normalize(dpdx), normalize(dpdy)));
    
    // Create a PBR material
    PBRMaterial material;
    material.albedo = in.color.rgb;
    
    // Vary roughness and metallic based on surface properties
    // float distanceFromOrigin = length(in.worldPosition); // Unused
    material.roughness = clamp(in.color.r * 0.5 + 0.2, kMinRoughness, 1.0);
    material.metallic = in.color.g * 0.8;
    material.ambientOcclusion = 1.0;
    material.emissive = float3(0.0);
    
    // Add emissive effect for certain parts
    if (uniforms.options.z > 0) {  // Special effects enabled
        // Make parts glow based on w-coordinate
        float glowFactor = sin(in.wComponent * 3.14159 + uniforms.time) * 0.5 + 0.5;
        material.emissive = getRainbowColor(in.wComponent + uniforms.time * 0.1) * glowFactor * 0.5;
    }
    
    // Lighting calculation
    float3 N = normalize(in.normal);
    float3 V = normalize(in.viewDirection);
    float3 L = normalize(in.lightDirection);
    float3 H = normalize(L + V);
    
    // Advanced lighting setup
    float3 lightColor = float3(1.0);
    float3 ambientColor = float3(0.1, 0.1, 0.2); // Slightly blue ambient
    
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    // float NdotH = max(dot(N, H), 0.0); // Unused
    
    // Calculate final color based on render mode
    float4 finalColor;
    
    if (uniforms.options.x > 0) {
        // Wireframe rendering with enhanced effects
        finalColor = in.color;
        
        // Pulse the wireframe brightness
        float glowIntensity = sin(uniforms.time * 2.0) * 0.3 + 1.2;
        finalColor.rgb *= glowIntensity;
        
        // Add edge highlighting based on viewing angle
        float edgeFactor = 1.0 - abs(dot(N, V));
        float edgeGlow = smoothstep(0.3, 1.0, edgeFactor);
        finalColor.rgb = mix(finalColor.rgb, float3(0.3, 0.7, 1.0), edgeGlow * 0.7);
        
        // Add 4D coordinate visualization
        float wFactor = in.wComponent * 0.5 + 0.5; // Map to 0-1 range
        float3 wColor = getRainbowColor(wFactor + uniforms.time * 0.1);
        finalColor.rgb = mix(finalColor.rgb, wColor, 0.2);
        
        // Add anti-aliasing for wireframe edges
        float lineWidth = 0.99;
        if (edgeStrength > lineWidth) {
            float lineAntialias = smoothstep(lineWidth, 1.0, edgeStrength);
            finalColor.a *= (1.0 - lineAntialias);
        }
        
    } else {
        // PBR lighting calculation
        // Base reflectivity for dielectrics is 0.04, for metals it's the albedo color
        float3 F0 = mix(float3(0.04), material.albedo, material.metallic);
        
        // Cook-Torrance BRDF calculation
        float NDF = distributionGGX(N, H, material.roughness);
        float G = geometrySmith(N, V, L, material.roughness);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        float3 numerator = NDF * G * F;
        float denominator = 4.0 * max(NdotV, 0.0) * max(NdotL, 0.0) + 0.0001;
        float3 specular = numerator / denominator;
        
        float3 kS = F;  // Energy of light that gets reflected
        float3 kD = (float3(1.0) - kS) * (1.0 - material.metallic); // Energy of light that gets refracted
        
        float3 diffuse = kD * material.albedo / PI;
        
        // Combine lighting components
        float3 Lo = (diffuse + specular) * lightColor * NdotL;
        
        // Add ambient, rim lighting, and emissive
        float3 ambient = ambientColor * material.albedo * material.ambientOcclusion;
        
        // Rim lighting effect (stronger at grazing angles)
        float rim = pow(1.0 - max(dot(N, V), 0.0), kRimPower) * kRimIntensity;
        float3 rimColor = lightColor * rim;
        
        float3 color = ambient + Lo + material.emissive + rimColor;
        
        // HDR tonemapping and gamma correction
        color = color / (color + float3(1.0)); // Reinhard tone mapping
        color = pow(color, float3(1.0/2.2));  // Gamma correction
        
        finalColor = float4(color, 1.0);
        
        // Add depth-based effects
        float depth = in.projPosition.z / in.projPosition.w;
        float depthFade = smoothstep(0.95, 1.0, depth);
        finalColor.rgb = mix(finalColor.rgb, float3(0.0, 0.0, 0.2), depthFade * 0.7); // Fade to dark blue at edges
        
        // Add subtle visual noise for more organic look
        float noise = fract(sin(dot(in.worldPosition.xy, float2(12.9898, 78.233))) * 43758.5453);
        finalColor.rgb += (noise - 0.5) * 0.02;
    }
    
    // Edge detection visualization if showing normals
    if (uniforms.options.y > 0) {
        float edgeHighlight = smoothstep(0.1, 0.3, edgeStrength);
        finalColor.rgb = mix(finalColor.rgb, float3(1.0, 1.0, 0.0), edgeHighlight * 0.7);
        
        // Add normal visualization
        finalColor.rgb = mix(finalColor.rgb, (N * 0.5 + 0.5), 0.5);
    }
    
    // Add a subtle vignette effect
    float2 uv = in.position.xy / float2(1024, 768); // Approximate screen size
    uv = (uv - 0.5) * 2.0;
    float vignette = 1.0 - dot(uv, uv) * 0.25;
    vignette = smoothstep(0.0, 1.0, vignette);
    finalColor.rgb *= vignette;
    
    // Ensure color is visible
    finalColor.rgb = max(finalColor.rgb, 0.05); // Prevent colors from getting too dark
    
    return finalColor;
}

// Additional shader for normal visualization
vertex VertexOut normalVisualizationVertex(const Vertex3D in [[stage_in]],
                                          constant Uniforms &uniforms [[buffer(1)]],
                                          uint vertexID [[vertex_id]]) {
    // Only used when showing normals option is enabled
    
    // Extract base position and normal
    float3 position = in.position;
    float3 normal = normalize(in.normal);
    
    // Scale the normal for visualization
    float normalScale = 0.2;
    
    // Generate two vertices: base and tip
    bool isTip = vertexID % 2 == 1;
    float3 vertexPosition = isTip ? position + normal * normalScale : position;
    
    // Transform to world space
    float4 worldPosition = uniforms.modelMatrix * float4(vertexPosition, 1.0);
    float4 viewPosition = uniforms.viewMatrix * worldPosition;
    float4 clipPosition = uniforms.projectionMatrix * viewPosition;
    
    // Transform normal to world space
    float3 worldNormal = normalize(uniforms.normalMatrix * normal);
    
    // Set color: base of normal is blue, tip is yellow
    float4 color = isTip ? float4(1.0, 1.0, 0.0, 1.0) : float4(0.0, 0.0, 1.0, 1.0);
    
    // Prepare output
    VertexOut out;
    out.position = clipPosition;
    out.projPosition = clipPosition;
    out.worldPosition = worldPosition.xyz;
    out.normal = worldNormal;
    out.color = color;
    out.texCoord = float2(0.0);
    out.viewDirection = float3(0.0);
    out.lightDirection = float3(0.0);
    out.wComponent = 0.0;
    
    return out;
}

// Additional fragment shader for post-processing
fragment float4 postProcessingShader(VertexOut in [[stage_in]],
                                    texture2d<float> colorTexture [[texture(0)]],
                                    texture2d<float> depthTexture [[texture(1)]],
                                    constant Uniforms &uniforms [[buffer(1)]]) {
    constexpr sampler textureSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
    
    float2 texCoord = in.texCoord;
    float4 color = colorTexture.sample(textureSampler, texCoord);
    float depth = depthTexture.sample(textureSampler, texCoord).r;
    
    // Apply post-processing effects based on depth and time
    
    // Chromatic aberration at edges
    float2 texCoordR = texCoord + float2(0.002, 0.000) * depth;
    float2 texCoordB = texCoord - float2(0.002, 0.000) * depth;
    
    float4 colorR = colorTexture.sample(textureSampler, texCoordR);
    float4 colorB = colorTexture.sample(textureSampler, texCoordB);
    
    color.r = mix(color.r, colorR.r, 0.5);
    color.b = mix(color.b, colorB.b, 0.5);
    
    // Add subtle bloom effect
    float2 texOffset = 1.0 / float2(1024, 768); // Approximate texture size
    float4 bloom = float4(0.0);
    
    // Sample 5x5 blur kernel
    for(int y = -2; y <= 2; y++) {
        for(int x = -2; x <= 2; x++) {
            float2 offset = float2(float(x), float(y)) * texOffset;
            bloom += colorTexture.sample(textureSampler, texCoord + offset) * 0.04;
        }
    }
    
    // Only apply bloom to bright areas
    float luminance = dot(color.rgb, float3(0.299, 0.587, 0.114));
    float bloomFactor = smoothstep(0.7, 1.0, luminance);
    color = mix(color, bloom, bloomFactor * 0.3);
    
    // Time-based color grading
    float timeFactor = sin(uniforms.time * 0.1) * 0.5 + 0.5;
    float3 tint = mix(float3(1.0, 0.9, 0.8), float3(0.8, 0.9, 1.0), timeFactor);
    color.rgb *= tint;
    
    return color;
}
