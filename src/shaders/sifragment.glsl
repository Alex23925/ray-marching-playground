#include <packing>

varying vec2 vUv;
varying vec3 hitPos;
varying vec4 vWorldPos;
varying vec3 viewZ;
uniform mat4 modelMatrix;
uniform mat4 uInverseWorldMatrix;
uniform float uTime;

// float readDepth( sampler2D depthSampler, vec2 coord ) {
// 				float fragCoordZ = texture2D( depthSampler, coord ).x;
// 				float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
// 				return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );
// }

void main() {
    
    float fogRadius = 1.0;
    vec4 fogCenter = vec4(0.0, 0.0, 0.0, fogRadius);
    vec3 fogColor = vec3(1.0);
    float innerRatio = .5;
    float density = .5;

    gl_FragColor = vec4(fogColor, 1.0);
}

// float calculateFogIntensity(
//     vec3 sphereCenter,
//     float sphereRadius,
//     float innerRatio,
//     float density,
//     vec3 cameraPos,
//     vec3 viewDir,
//     float maxDistance
// ) {
//     // calcualte ray-sphere intersection
//     vec3 localCam = cameraPos - sphereCenter;
//     float a = dot(viewDir, viewDir);
//     float b = 2.0 * dot(cameraPos, viewDir);
//     float c = dot(localCam, localCam) - sphereRadius * sphereRadius;
//     float d = b * b - 4.0 * a * c;
    
//     if( d <= 0.0) {
//         return 0;
//     }
    
//     float Dsqrt = sqrt(d);
//     float dist = max((-b - Dsqrt)/2.0*a, 0.0);
//     float dist2 = max((-b + Dsqrt)/2.0*a, 0.0);

//     float backDepth = min(maxDistance, dist2);
//     float mySample = dist;
//     float step_distance = (backDepth - dist1)/10.0;
//     float step_contribution = density;

//     float centerValue = 1.0/(1.0 - innerRatio);

//     float clarity = 1.0;

//     for(int i = 0.0; i < 10.0; i ++) {
//         vec3 pos = localCam + viewDir * mySample;
//         float val = clamp(centerValue * (1.0 - length(pos)/sphereRadius), 0.0, 1.0);
//         float fog_amount = clamp(val * step_contribution, 0.0, 1.0);
//         clarity *= (1.0 - fog_amount);
//         mySample += step_distance;
//     }
    

//     return 1 - clarity;
// }
