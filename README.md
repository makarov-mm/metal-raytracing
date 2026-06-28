# Metal Hardware Ray Tracing — glass scene

A Swift + Metal port of the C++/Vulkan hardware ray tracer. A central morphing
supertoroid in clear glass, ringed by six coloured glass shapes (sphere,
diamond, cube, square pyramid, cylinder, regular dodecahedron). Dispersive
Fresnel glass, soft area-light shadows, and per-pixel adaptive temporal
accumulation.

## Requirements

- macOS with a Metal **ray-tracing capable** GPU (Apple silicon M-series, or a
  recent AMD). The app checks `device.supportsRaytracing` at start.
- Xcode command-line tools (`swiftc`, the Metal toolchain).

## Build & run

```sh
chmod +x build.sh
./build.sh
./RayTracer
```

`Shaders.metal` is compiled at runtime, so keep it in the same folder as the
`RayTracer` binary. (No Xcode project or `.metallib` step needed.)

## Controls

| Input        | Action      |
|--------------|-------------|
| Mouse drag   | Orbit       |
| Scroll       | Zoom         |
| R            | Reset view  |
| F            | Fullscreen  |
| ESC          | Quit        |

## How it maps to the Vulkan version

- **Acceleration structures.** One primitive AS (all geometry in a single
  vertex/index buffer) plus a one-instance instance AS — same single-BLAS design
  as the Vulkan build. The supertoroid deforms every frame, so the primitive AS
  is **refit** each frame and the instance AS rebuilt (it is tiny).
- **Glass.** Per-channel IOR dispersion, Schlick Fresnel, coloured transmission
  tint. Metal traces iteratively in a compute kernel (no recursive hit shaders),
  so at each glass hit it picks reflect-or-refract **stochastically by the
  Fresnel term** instead of splitting into two recursive rays. With the heavy
  accumulation this converges to the same image as the Vulkan Whitted split.
- **Soft shadows.** Area light sampled with a cone of shadow rays (any-hit
  intersector), identical penumbra logic.
- **Adaptive accumulation.** Two persistent `rgba32f` textures: one accumulates
  colour + per-pixel sample count, the other stores the previous frame's
  non-jittered primary-hit position. Static pixels accumulate forever (converge
  to clean, sharp glass); pixels whose surface moved this frame reset, so the
  spinning/morphing geometry stays sharp. Camera motion resets the whole buffer.

## Tuning knobs

- `U.params` in `Renderer.draw` — `(time, maxBounces, lightIntensity, spp)`.
  Raise `spp` (currently 12) for cleaner moving glass at the cost of FPS.
- Motion-detection threshold `0.0008 * camDist + 0.0012` in `Shaders.metal`.
- Glass colour strength: the `0.35` in the `mix(float3(1.0), h.albedo, 0.35)`
  line in `Shaders.metal`.
- Supertoroid spiral: the integer `twist` (= 3) in `Renderer.updateTorus`. Keep
  it an integer or the ring tears at the seam.

## Note

This was written without an on-hand Metal toolchain to compile against, so treat
the first build as the shakedown run. The shading math, geometry, and
accumulation mirror the verified Vulkan version; the Metal ray-tracing API glue
(acceleration-structure build/refit, intersector setup) is the part most worth
sanity-checking against your SDK version if anything does not link.
