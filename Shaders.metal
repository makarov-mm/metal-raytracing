//  Shaders.metal
//  Hardware ray tracing in a Metal compute kernel — a faithful port of the
//  Vulkan renderer's shading (dispersive glass, soft area-light shadows,
//  per-pixel adaptive temporal accumulation).
//
//  Metal traces iteratively in a compute kernel (no recursive hit shaders),
//  so glass picks reflect-or-refract stochastically by the Fresnel term; with
//  the heavy accumulation this converges to the same image as the Vulkan
//  Whitted split.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// 48-byte vertex, identical layout to the C++/Vulkan side:
//   position (12) | reflectivity (4) | normal (12) | matId (4) | color (12) | pad (4)
struct Vertex {
    packed_float3 position;
    float         reflectivity;
    packed_float3 normal;
    float         matId;        // 0 = floor, 1 = solid, 2 = glass, 3 = emissive
    packed_float3 color;
    float         pad;
};

struct Uniforms {
    float4x4 viewInverse;
    float4x4 projInverse;
    float4   lightPos;   // xyz = position, w = radius
    float4   params;     // x = time, y = maxBounces, z = intensity, w = spp
    uint2    frame;      // x = accumulation index (0 on camera move), y = free-running
    uint2    dim;        // width, height
};

constant float IOR_R = 1.50, IOR_G = 1.52, IOR_B = 1.54;

// ---------------------------------------------------------------------------
//  RNG + helpers
// ---------------------------------------------------------------------------
inline uint pcg(thread uint& state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline float rnd(thread uint& state) { return float(pcg(state)) * (1.0 / 4294967296.0); }

inline float3 skyColor(float3 d) {
    float h = 0.5 * (normalize(d).y + 1.0);
    return mix(float3(0.90, 0.95, 1.00), float3(0.25, 0.45, 0.85), h);
}
inline float3 checker(float3 p) {
    float s = floor(p.x) + floor(p.z);
    return mix(float3(0.85), float3(0.10), fmod(s, 2.0));
}
inline void basis(float3 n, thread float3& t, thread float3& b) {
    float3 a = abs(n.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    t = normalize(cross(n, a));
    b = cross(n, t);
}
inline float3 sampleCone(float3 dir, float cosThetaMax, thread uint& seed) {
    float u1 = rnd(seed), u2 = rnd(seed);
    float cosT = mix(1.0, cosThetaMax, u1);
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi  = 6.28318530718 * u2;
    float3 t, b; basis(dir, t, b);
    return normalize(t * (cos(phi) * sinT) + b * (sin(phi) * sinT) + dir * cosT);
}

// Result of one closest-hit lookup (mirrors the Vulkan HitPayload).
struct Hit {
    float3 pos;
    float3 normal;      // faces against the incoming ray
    float3 albedo;
    float  reflectivity;
    int    mat;
    bool   front;       // true = hit the outer surface
    bool   miss;
};

inline Hit closestHit(ray r,
                      instance_acceleration_structure accel,
                      device const Vertex* verts,
                      device const uint* idx) {
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);

    intersection_result<triangle_data, instancing> res = isect.intersect(r, accel, 0xFF);

    Hit h;
    if (res.type == intersection_type::none) { h.miss = true; return h; }
    h.miss = false;

    uint p = res.primitive_id;             // one geometry -> global triangle index
    uint i0 = idx[3 * p + 0], i1 = idx[3 * p + 1], i2 = idx[3 * p + 2];
    Vertex a = verts[i0], b = verts[i1], c = verts[i2];

    float2 bc = res.triangle_barycentric_coord;
    float3 bary = float3(1.0 - bc.x - bc.y, bc.x, bc.y);

    float3 N = normalize(float3(a.normal) * bary.x + float3(b.normal) * bary.y + float3(c.normal) * bary.z);
    h.albedo       = float3(a.color) * bary.x + float3(b.color) * bary.y + float3(c.color) * bary.z;
    h.reflectivity = a.reflectivity;
    h.mat          = int(a.matId + 0.5);

    bool front = dot(N, r.direction) < 0.0;
    if (!front) N = -N;
    h.normal = N;
    h.front  = front;
    h.pos    = r.origin + r.direction * res.distance;
    return h;
}

inline bool occluded(ray r,
                     instance_acceleration_structure accel) {
    intersector<triangle_data, instancing> shad;
    shad.assume_geometry_type(geometry_type::triangle);
    shad.force_opacity(forced_opacity::opaque);
    shad.accept_any_intersection(true);     // stop at the first hit
    auto res = shad.intersect(r, accel, 0xFF);
    return res.type != intersection_type::none;
}

inline ray makeRay(float3 o, float3 d, float tmin, float tmax) {
    ray r; r.origin = o; r.direction = d; r.min_distance = tmin; r.max_distance = tmax; return r;
}

// Camera ray for a pixel (with optional sub-pixel offset).
inline float3 cameraRay(constant Uniforms& U, float2 pixel, thread float3& origin) {
    float2 inUV = pixel / float2(U.dim);
    float2 d    = inUV * 2.0 - 1.0;
    d.y = -d.y;                 // Metal drawable origin is top-left -> flip vertical
    origin      = (U.viewInverse * float4(0, 0, 0, 1)).xyz;
    float4 tgt  = U.projInverse * float4(d.x, d.y, 1, 1);
    return normalize((U.viewInverse * float4(normalize(tgt.xyz), 0)).xyz);
}

// ---------------------------------------------------------------------------
//  One full path for one (jittered) sample.
// ---------------------------------------------------------------------------
inline float3 renderSample(constant Uniforms& U,
                           instance_acceleration_structure accel,
                           device const Vertex* verts,
                           device const uint* idx,
                           uint2 gid, int sampleIdx, thread uint& seed) {
    float2 jitter = float2(rnd(seed), rnd(seed)) - 0.5;
    float3 origin;
    float3 dir = cameraRay(U, float2(gid) + 0.5 + jitter, origin);

    int   maxBounces = int(U.params.y);
    float intensity  = U.params.z;
    const float eps  = 0.001;

    float3 color = float3(0.0), throughput = float3(1.0);
    int  hero = sampleIdx % 3;          // stratified spectral channel
    bool dispersed = false;

    for (int bounce = 0; bounce <= maxBounces; ++bounce) {
        Hit h = closestHit(makeRay(origin, dir, 0.001, 10000.0), accel, verts, idx);
        if (h.miss) { color += throughput * skyColor(dir); break; }

        float3 N = h.normal;

        if (h.mat == 3) {                                   // emissive light
            color += throughput * h.albedo * 3.0;
            break;
        }

        if (h.mat == 2) {                                   // dispersive glass
            float ior = (hero == 0) ? IOR_R : (hero == 1) ? IOR_G : IOR_B;
            float eta = h.front ? (1.0 / ior) : ior;

            float cosI = clamp(dot(-dir, N), 0.0, 1.0);
            float R0 = (1.0 - 1.52) / (1.0 + 1.52); R0 *= R0;
            float fres = R0 + (1.0 - R0) * pow(1.0 - cosI, 5.0);

            float3 refr = refract(dir, N, eta);
            bool tir = dot(refr, refr) < 1e-6;

            if (tir || rnd(seed) < fres) {
                dir    = reflect(dir, N);
                origin = h.pos + N * eps;
            } else {
                dir    = normalize(refr);
                origin = h.pos - N * eps;
                if (!dispersed) {
                    float3 mask = (hero == 0) ? float3(1,0,0)
                                : (hero == 1) ? float3(0,1,0) : float3(0,0,1);
                    throughput *= 3.0 * mask;
                    dispersed = true;
                }
                throughput *= mix(float3(1.0), h.albedo, 0.35);   // coloured glass
            }
            throughput *= float3(0.98);
            continue;
        }

        // floor / solid
        float3 albedo = (h.mat == 0) ? checker(h.pos) : h.albedo;

        float3 toLight = U.lightPos.xyz - h.pos;
        float dist = length(toLight);
        float3 Lc = toLight / dist;
        float sinMax = clamp(U.lightPos.w / dist, 0.0, 0.999);
        float cosMax = sqrt(1.0 - sinMax * sinMax);
        float tmax = dist - U.lightPos.w * 1.001;

        const int SHADOW_SAMPLES = 6;
        float diffAcc = 0.0;
        for (int si = 0; si < SHADOW_SAMPLES; ++si) {
            float3 Ls = sampleCone(Lc, cosMax, seed);
            bool occ = occluded(makeRay(h.pos + N * eps, Ls, 0.001, tmax), accel);
            diffAcc += max(dot(N, Ls), 0.0) * (occ ? 0.0 : 1.0);
        }
        float diff = (diffAcc / float(SHADOW_SAMPLES)) * intensity;
        float3 localc = albedo * (0.12 + diff);

        float refl = h.reflectivity;
        color += throughput * localc * (1.0 - refl);
        if (refl <= 0.001) break;
        throughput *= refl;
        dir    = reflect(dir, N);
        origin = h.pos + N * eps;
    }
    return color;
}

// Non-jittered centre primary hit position, for the motion test only.
inline float3 primaryHitPos(constant Uniforms& U,
                            instance_acceleration_structure accel,
                            device const Vertex* verts,
                            device const uint* idx, uint2 gid) {
    float3 origin;
    float3 dir = cameraRay(U, float2(gid) + 0.5, origin);
    Hit h = closestHit(makeRay(origin, dir, 0.001, 10000.0), accel, verts, idx);
    if (h.miss) return origin + dir * 1000.0;
    return h.pos;
}

// ---------------------------------------------------------------------------
//  Main kernel
// ---------------------------------------------------------------------------
kernel void rtKernel(uint2 gid [[thread_position_in_grid]],
                     constant Uniforms& U [[buffer(0)]],
                     instance_acceleration_structure accel [[buffer(1)]],
                     device const Vertex* verts [[buffer(2)]],
                     device const uint* idx [[buffer(3)]],
                     texture2d<float, access::read_write> accumTex [[texture(0)]],
                     texture2d<float, access::read_write> histTex  [[texture(1)]],
                     texture2d<float, access::write>      outTex   [[texture(2)]]) {
    if (gid.x >= U.dim.x || gid.y >= U.dim.y) return;

    int spp = max(1, int(U.params.w));
    float3 color = float3(0.0);
    for (int s = 0; s < spp; ++s) {
        uint seed = (gid.x * 1973u + gid.y * 9277u
                   + (U.frame.y * uint(spp) + uint(s)) * 26699u) | 1u;
        color += renderSample(U, accel, verts, idx, gid, s, seed);
    }
    color /= float(spp);

    // ---- per-pixel adaptive accumulation ----
    float3 curPos  = primaryHitPos(U, accel, verts, idx, gid);
    float4 prevH   = histTex.read(gid);
    float3 eye     = (U.viewInverse * float4(0, 0, 0, 1)).xyz;
    float camDist  = distance(curPos, eye);
    float moved    = distance(curPos, prevH.xyz);
    bool reset     = (U.frame.x == 0u) || (moved > 0.0008 * camDist + 0.0012);

    float4 prevA = accumTex.read(gid);
    float count  = reset ? 0.0 : prevA.a;
    float3 accum = (count < 0.5) ? color : mix(prevA.rgb, color, 1.0 / (count + 1.0));
    count += 1.0;
    accumTex.write(float4(accum, count), gid);
    histTex.write(float4(curPos, 0.0), gid);

    float3 outc = pow(clamp(accum, 0.0, 1.0), float3(1.0 / 2.2));
    outTex.write(float4(outc, 1.0), gid);
}
