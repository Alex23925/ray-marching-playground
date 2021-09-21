varying vec2 vUv;

varying vec3 hitPos;
varying vec4 vWorldPos;
varying vec3 vViewDir;
varying vec4 vClipPos;
varying vec3 viewZ;

vec4 computeScreenPos(vec4 clipPos) {
    vec4 o = clipPos * .5;
     return vec4(0.0);
}

void main()
{
    vec4 modelPosition = modelMatrix * vec4(position, 1.0);

    vec4 objToClipPos = projectionMatrix * modelViewMatrix * vec4(position, 1.0);

    gl_Position = objToClipPos;

    vUv = uv;
    hitPos = position; // object space
    vWorldPos = modelPosition; // world space
    vViewDir = modelPosition.xyz - cameraPosition;
    vClipPos = objToClipPos; // clip space
    viewZ = -(modelViewMatrix * vec4(position.xyz, 1.)).xyz;

}