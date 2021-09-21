varying vec2 vUv;
varying vec3 hitPos;
varying vec4 vWorldPos;

uniform mat4 modelMatrix;
uniform mat4 uInverseWorldMatrix;
uniform float uTime;
uniform sampler2D matcap;
uniform vec2 mouse;
uniform float uSteps;
uniform float uIterations;
uniform float uPower;
uniform float uBailout;
uniform float uLightIntensity;
uniform float uSinScale;
uniform float uGlowBrightness;
uniform float uGlowThickness;

uniform float uColor1R;
uniform float uColor1G;
uniform float uColor1B;

uniform float uColor2R;
uniform float uColor2G;
uniform float uColor2B;

#define MAX_STEPS 100.0
#define MAX_DIST 100.0
#define SURF_DIST .001
#define PI 3.1415926535897932384626433832795

//	Simplex 3D Noise 
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){ 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //  x0 = x0 - 0. + 0.0 * C 
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;

    // Permutations
    i = mod(i, 289.0 ); 
    vec4 p = permute( permute( permute( 
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                    dot(p2,x2), dot(p3,x3) ) );
}

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

vec3 rotate(vec3 v, vec3 axis, float angle) {
	mat4 m = rotationMatrix(axis, angle);
	return (m * vec4(v, 1.0)).xyz;
}

vec2 getmatcap(vec3 eye, vec3 normal) {
  vec3 reflected = reflect(eye, normal);
  float m = 2.8284271247461903 * sqrt( reflected.z+1.0 );
  return reflected.xy / m + 0.5;
}

float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float sdSphere(vec3 p, float r) {
    float d = length(p) - r;

    return d;
}

float sdTorus(vec3 p) {
    float d = length(vec2(length(p.xz)-.1, p.y)) - .1;

    return d;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdBoxFrame( vec3 p, vec3 b, float e )
{
  p = abs(p  )-b;
  vec3 q = abs(p+e)-e;
  return min(min(
      length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
      length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
      length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

float sdOctahedron( vec3 p, float s)
{
  p = abs(p);
  return (p.x+p.y+p.z-s)*0.57735027;
}


float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float sdBulb(vec3 pos) {
	vec3 z = pos;
	float dr = 1.0;
	float r = 0.0;
    // abs(sin(uPower + uTime/5.0) * 10.0)
    float power = abs(sin(uPower + uTime/5.0) * 10.0);
	for (float i = 0.0; i < uIterations; i++) {
		r = length(z);
		if (r>uBailout) break;
		
		// convert to polar coordinates
		float theta = acos(z.z/r);
		float phi = atan(z.y,z.x);
		dr =  pow( r, power-1.0)*power*dr + 1.0;
		
		// scale and rotate the point
		float zr = pow( r,power);
		theta = theta*power ;
		phi = phi*power;
		
		// convert back to cartesian coordinates
		z = zr*vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
		z+=pos;
	}
	return 0.5*log(r)*r/dr;
}

float sinCrazy(vec3 p) {
    return 1.0 - (cos(p.x) + sin(p.y) + sin(p.z))/3.0;
}

float squareSin(vec3 p) {
    return uSinScale + uSinScale*smoothstep(0.0,0.7,1.0 - (sin(p.x) + sin(p.y) + sin(p.z))/3.0);
}

float plasma(vec3 p){
    return 1.0 - (sin(p.x) + sin(snoise(p + uTime)) + cos(p.z))/3.0;
}


float scene(vec3 p) {
    vec3 p1 = rotate(p, vec3(1.0), uTime/.5);

    float sphere = sdSphere(p, .5);
    float box = sdBox(p, vec3(.4));
    float bulb = sdBulb(p);

    float scale = uSinScale + uSinScale * sin(uTime/6.0);

    //  max(sphere, (0.85 - sinCrazy(p1*scale))/scale)

    return max(sphere, (0.85 - sinCrazy(p1*scale))/scale);
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(.01, 0.0);
    vec3 n = scene(p) - vec3(
        scene(p-e.xxy),
        scene(p-e.yxy),
        scene(p-e.yyx)
    );

    return normalize(n);
}

vec3 getColor(float amount) {
    vec3 col = .5 + .5 * cos(6.28319 * (vec3(0.2, 0.0, 0.0) + amount * vec3(1.0, 1.0, .5)));
    return col * amount;
}

vec3 getColorAmount(vec3 p) {
    float amount = clamp( (1.5 - length(p))/2.0, 0.0, 1.0);
   
    vec3 col = uGlowBrightness + uGlowThickness * cos((2.0*PI)* (vec3(0.2, 0.0, 0.0) + amount * vec3(1.0, 1.0, .5)));
    //vec3 col = uGlowBrightness + uGlowThickness * cos(6.28319 * (vec3(uColor1R, uColor1G, uColor1B) + amount * vec3(uColor2R, uColor2G, uColor2B)));
    return col * amount;
}

vec3 rayPos;
vec3 color = vec3(0.0);

float raymarch(vec3 ro, vec3 rd) {

    float dO = 0.0;
    float dS;

    for (float i = 0.0; i < uSteps; i++) {
        vec3 p = ro + dO * rd;
        rayPos = p;
        dS = scene(p);
        dO += dS;
        if( abs(dS) < SURF_DIST || dO > uSteps ) {break;};

        if ( dO < MAX_DIST ) {
            color += .02*getColorAmount(rayPos);
        }
    }
    
    return dO;
}


float sphereIntersection() {
    return 0.0;
}

void main() {

    float dist = length(vUv - vec2(.5));
    vec3 bg = mix(vec3(0.3), vec3(0.0), dist);

    vec3 light = vec3(1.0);
    vec3 lightColor = vec3(0.302, 0.0157, 0.6784);

    vec2 uv = vUv - .5;
   
    vec4 ro = inverse(modelMatrix) * vec4(cameraPosition,  1.0);
    vec3 rd = normalize(hitPos - ro.xyz);

    // shader camera and hit pos
    vec3 ro2 = vec3(0, 0, -3);
    vec3 rd2 = normalize(vec3(uv.x, uv.y, 1.0));

    float d = raymarch(cameraPosition, rd);

     if ( d < MAX_DIST ) {
         vec3 p = ro.xyz + rd * d;
         vec3 n = getNormal(p);
         float diff = dot(n, light);
        vec2 matcapUV = getmatcap(rd, n);

        // color.rgb = vec3(diff);
        float fresnel = pow( 1.0 + dot( rd, n), 3.0 );
        //color = vec3(fresnel);

        //color = texture2D(matcap, matcapUV).rgb;
        diff *= uLightIntensity;

        color = color;   
    }
    gl_FragColor = vec4(vec3(1.0), 0.0);
}


