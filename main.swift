// main.swift
// Hardware ray-traced glass scene — Swift + Metal (MetalKit / MTKView), macOS.
//
// A full analog of the C++/Vulkan renderer: a central morphing supertoroid in
// clear glass, ringed by six coloured glass shapes (sphere, diamond, cube,
// pyramid, cylinder, regular dodecahedron). Dispersive Fresnel glass, soft
// area-light shadows, and per-pixel adaptive temporal accumulation.
//
// Requires a Metal ray-tracing capable GPU (Apple silicon, or recent AMD).
//
// Build (single file + Shaders.metal, no Xcode project) — see build.sh:
//   ./build.sh && ./RayTracer
//
// Controls:
//   Mouse drag — orbit      Scroll — zoom
//   R — reset view          F — fullscreen      ESC — quit

import Cocoa
import MetalKit
import simd

// ============================================================
// MARK: - Math
// ============================================================

func perspective(_ fovY: Float, _ aspect: Float, _ near: Float, _ far: Float) -> simd_float4x4 {
    let f = 1.0 / tanf(fovY * 0.5)
    return simd_float4x4(columns: (
        SIMD4<Float>(f / aspect, 0, 0, 0),
        SIMD4<Float>(0, f, 0, 0),
        SIMD4<Float>(0, 0, far / (near - far), -1),
        SIMD4<Float>(0, 0, (far * near) / (near - far), 0)
    ))
}

func lookAt(_ eye: SIMD3<Float>, _ center: SIMD3<Float>, _ up: SIMD3<Float>) -> simd_float4x4 {
    let z = normalize(eye - center)          // right-handed: camera looks down -z
    let x = normalize(cross(up, z))
    let y = cross(z, x)
    return simd_float4x4(columns: (
        SIMD4<Float>(x.x, y.x, z.x, 0),
        SIMD4<Float>(x.y, y.y, z.y, 0),
        SIMD4<Float>(x.z, y.z, z.z, 0),
        SIMD4<Float>(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
    ))
}

// ============================================================
// MARK: - Shared types (must match Shaders.metal byte-for-byte)
// ============================================================

// 48-byte vertex: position | reflectivity | normal | matId | color | pad
struct Vertex {
    var px: Float = 0, py: Float = 0, pz: Float = 0, reflectivity: Float = 0
    var nx: Float = 0, ny: Float = 1, nz: Float = 0, matId: Float = 0
    var cx: Float = 0, cy: Float = 0, cz: Float = 0, pad: Float = 0
}

struct Uniforms {
    var viewInverse = matrix_identity_float4x4
    var projInverse = matrix_identity_float4x4
    var lightPos = SIMD4<Float>(0, 0, 0, 0)
    var params   = SIMD4<Float>(0, 0, 0, 0)
    var frame    = SIMD2<UInt32>(0, 0)
    var dim      = SIMD2<UInt32>(0, 0)
}

// ============================================================
// MARK: - Geometry builder
// ============================================================

final class Mesh {
    var verts: [Vertex] = []
    var idx: [UInt32] = []

    func push(_ p: SIMD3<Float>, _ n: SIMD3<Float>, _ c: SIMD3<Float>, _ refl: Float, _ mat: Float) {
        var v = Vertex()
        v.px = p.x; v.py = p.y; v.pz = p.z; v.reflectivity = refl
        v.nx = n.x; v.ny = n.y; v.nz = n.z; v.matId = mat
        v.cx = c.x; v.cy = c.y; v.cz = c.z
        verts.append(v)
    }

    func tri(_ a: UInt32, _ b: UInt32, _ c: UInt32) { idx.append(a); idx.append(b); idx.append(c) }

    // ---- primitives -------------------------------------------------------
    func addFloor(_ s: Float, _ y: Float) {
        let n = SIMD3<Float>(0, 1, 0), col = SIMD3<Float>(1, 1, 1)
        let b = UInt32(verts.count)
        push(SIMD3(-s, y, -s), n, col, 0, 0); push(SIMD3(s, y, -s), n, col, 0, 0)
        push(SIMD3(s, y, s), n, col, 0, 0);   push(SIMD3(-s, y, s), n, col, 0, 0)
        tri(b, b + 1, b + 2); tri(b, b + 2, b + 3)
    }

    func addSphere(_ center: SIMD3<Float>, _ r: Float, _ col: SIMD3<Float>, _ refl: Float, _ mat: Float,
                   _ stacks: Int = 48, _ slices: Int = 96) {
        let base = UInt32(verts.count)
        for i in 0...stacks {
            let v = Float(i) / Float(stacks), phi = v * Float.pi
            for j in 0...slices {
                let u = Float(j) / Float(slices), theta = u * 2 * Float.pi
                let n = SIMD3<Float>(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta))
                push(center + n * r, n, col, refl, mat)
            }
        }
        let row = UInt32(slices + 1)
        for i in 0..<UInt32(stacks) {
            for j in 0..<UInt32(slices) {
                let a = base + i * row + j, b = base + (i + 1) * row + j
                tri(a, b, a + 1); tri(a + 1, b, b + 1)
            }
        }
    }

    // Flat-shaded triangle with an outward normal (oriented away from interior).
    func flatTri(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, _ interior: SIMD3<Float>,
                 _ col: SIMD3<Float>, _ mat: Float) {
        var nn = normalize(cross(b - a, c - a))
        let cen = (a + b + c) / 3
        if dot(nn, cen - interior) < 0 { nn = -nn }
        let t = UInt32(verts.count)
        push(a, nn, col, 0, mat); push(b, nn, col, 0, mat); push(c, nn, col, 0, mat)
        tri(t, t + 1, t + 2)
    }

    func addCube(_ center: SIMD3<Float>, _ s: Float, _ col: SIMD3<Float>, _ mat: Float) {
        let nrm: [SIMD3<Float>] = [SIMD3(1,0,0),SIMD3(-1,0,0),SIMD3(0,1,0),SIMD3(0,-1,0),SIMD3(0,0,1),SIMD3(0,0,-1)]
        let h = s * 0.5
        for N in nrm {
            var u: SIMD3<Float>, w: SIMD3<Float>
            if abs(N.y) > 0.5 { u = SIMD3(1,0,0); w = SIMD3(0,0,1) } else { u = SIMD3(0,1,0); w = cross(N, u) }
            let c = N * h
            let p0 = c - u*h - w*h, p1 = c + u*h - w*h, p2 = c + u*h + w*h, p3 = c - u*h + w*h
            let b = UInt32(verts.count)
            push(p0 + center, N, col, 0, mat); push(p1 + center, N, col, 0, mat)
            push(p2 + center, N, col, 0, mat); push(p3 + center, N, col, 0, mat)
            tri(b, b + 1, b + 2); tri(b, b + 2, b + 3)
        }
    }

    func addDiamond(_ center: SIMD3<Float>, _ s: Float, _ col: SIMD3<Float>, _ mat: Float, _ N: Int = 16) {
        let rt: Float = 0.50, ht: Float = 0.42, rg: Float = 1.00, hp: Float = 1.15
        let c0 = SIMD3<Float>(0, (ht - hp) * 0.5 * s, 0)
        func T(_ i: Int) -> SIMD3<Float> { let a = Float((i % N + N) % N) * 2 * .pi / Float(N)
            return SIMD3(rt * cosf(a) * s, ht * s, rt * sinf(a) * s) }
        func G(_ i: Int) -> SIMD3<Float> { let a = (Float((i % N + N) % N) + 0.5) * 2 * .pi / Float(N)
            return SIMD3(rg * cosf(a) * s, 0, rg * sinf(a) * s) }
        let Tc = SIMD3<Float>(0, ht * s, 0), C = SIMD3<Float>(0, -hp * s, 0)
        for i in 0..<N {
            flatTri(Tc + center, T(i+1) + center, T(i) + center, center + c0, col, mat)
            flatTri(T(i) + center, T(i+1) + center, G(i) + center, center + c0, col, mat)
            flatTri(T(i+1) + center, G(i+1) + center, G(i) + center, center + c0, col, mat)
            flatTri(G(i) + center, G(i+1) + center, C + center, center + c0, col, mat)
        }
    }

    func addPyramid(_ center: SIMD3<Float>, _ s: Float, _ h: Float, _ col: SIMD3<Float>, _ mat: Float) {
        let hs = s * 0.5, interior = center + SIMD3<Float>(0, h * 0.30, 0)
        let b0 = center + SIMD3<Float>(-hs,0,-hs), b1 = center + SIMD3<Float>(hs,0,-hs)
        let b2 = center + SIMD3<Float>(hs,0,hs),  b3 = center + SIMD3<Float>(-hs,0,hs)
        let ap = center + SIMD3<Float>(0,h,0)
        flatTri(b0,b1,ap,interior,col,mat); flatTri(b1,b2,ap,interior,col,mat)
        flatTri(b2,b3,ap,interior,col,mat); flatTri(b3,b0,ap,interior,col,mat)
        flatTri(b0,b2,b1,interior,col,mat); flatTri(b0,b3,b2,interior,col,mat)
    }

    func addCylinder(_ center: SIMD3<Float>, _ r: Float, _ h: Float, _ col: SIMD3<Float>, _ mat: Float, _ N: Int = 56) {
        let top = center + SIMD3<Float>(0, h, 0)
        for i in 0..<N {
            let a0 = 2 * Float.pi * Float(i) / Float(N), a1 = 2 * Float.pi * Float(i + 1) / Float(N)
            let d0 = SIMD3<Float>(cosf(a0), 0, sinf(a0)), d1 = SIMD3<Float>(cosf(a1), 0, sinf(a1))
            let bl = center + d0*r, br = center + d1*r, tl = top + d0*r, tr = top + d1*r
            var b = UInt32(verts.count)
            push(bl, d0, col, 0, mat); push(br, d1, col, 0, mat); push(tr, d1, col, 0, mat); push(tl, d0, col, 0, mat)
            tri(b, b+1, b+2); tri(b, b+2, b+3)
            b = UInt32(verts.count)
            push(center, SIMD3(0,-1,0), col, 0, mat); push(center + d1*r, SIMD3(0,-1,0), col, 0, mat); push(center + d0*r, SIMD3(0,-1,0), col, 0, mat)
            tri(b, b+1, b+2)
            b = UInt32(verts.count)
            push(top, SIMD3(0,1,0), col, 0, mat); push(top + d0*r, SIMD3(0,1,0), col, 0, mat); push(top + d1*r, SIMD3(0,1,0), col, 0, mat)
            tri(b, b+1, b+2)
        }
    }

    func addDodecahedron(_ center: SIMD3<Float>, _ s: Float, _ col: SIMD3<Float>, _ mat: Float) {
        let P: Float = 1.6180339887, a: Float = 1.0 / 1.6180339887
        let DV: [SIMD3<Float>] = [
            SIMD3(-1,-1,-1),SIMD3(-1,-1,1),SIMD3(-1,1,-1),SIMD3(-1,1,1),
            SIMD3(1,-1,-1),SIMD3(1,-1,1),SIMD3(1,1,-1),SIMD3(1,1,1),
            SIMD3(0,-a,-P),SIMD3(0,-a,P),SIMD3(0,a,-P),SIMD3(0,a,P),
            SIMD3(-a,-P,0),SIMD3(-a,P,0),SIMD3(a,-P,0),SIMD3(a,P,0),
            SIMD3(-P,0,-a),SIMD3(-P,0,a),SIMD3(P,0,-a),SIMD3(P,0,a)
        ]
        let F: [[Int]] = [
            [14,12,0,8,4],[10,8,0,16,2],[17,16,0,12,1],[5,9,1,12,14],
            [3,17,1,9,11],[6,10,2,13,15],[3,13,2,16,17],[15,13,3,11,7],
            [6,18,4,8,10],[5,14,4,18,19],[11,9,5,19,7],[19,18,6,15,7]
        ]
        let scale = s / 1.7320508
        for f in F {
            let v = f.map { center + DV[$0] * scale }
            flatTri(v[0], v[1], v[2], center, col, mat)
            flatTri(v[0], v[2], v[3], center, col, mat)
            flatTri(v[0], v[3], v[4], center, col, mat)
        }
    }

    // ---- morphing supertoroid (regenerated every frame) -------------------
    // Reserves a (U+1)x(Vr+1) grid; returns its vertex offset.
    func reserveGrid(_ U: Int, _ Vr: Int, _ col: SIMD3<Float>, _ mat: Float) -> Int {
        let base = UInt32(verts.count)
        for _ in 0...(U) { for _ in 0...(Vr) { push(SIMD3(0,0,0), SIMD3(0,1,0), col, 0, mat) } }
        let stride = UInt32(Vr + 1)
        for i in 0..<UInt32(U) {
            for j in 0..<UInt32(Vr) {
                let aI = base + i * stride + j, bI = base + (i + 1) * stride + j
                tri(aI, bI, aI + 1); tri(aI + 1, bI, bI + 1)
            }
        }
        return Int(base)
    }
}

// ============================================================
// MARK: - Renderer
// ============================================================

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    var pipeline: MTLComputePipelineState!

    var mesh = Mesh()
    var vertexBuffer: MTLBuffer!
    var indexBuffer: MTLBuffer!
    let U = 120, Vr = 60
    var torusOffset = 0

    // acceleration structures
    var blas: MTLAccelerationStructure?
    var tlas: MTLAccelerationStructure?
    var blasScratch: MTLBuffer!
    var tlasScratch: MTLBuffer!
    var instanceBuffer: MTLBuffer!
    var triGeo = MTLAccelerationStructureTriangleGeometryDescriptor()
    var primDesc = MTLPrimitiveAccelerationStructureDescriptor()
    var instDesc = MTLInstanceAccelerationStructureDescriptor()

    var accumTex: MTLTexture?
    var histTex: MTLTexture?

    // camera (orbit)
    var azimuth: Float = 0.9
    var elevation: Float = 0.35
    var distance: Float = 16
    let target = SIMD3<Float>(0, 2.0, 0)

    var frameAccum: UInt32 = 0      // 0 on camera move -> resets accumulation
    var frameCounter: UInt32 = 0    // free-running (RNG)
    var needsReset = true
    let startTime = CACurrentMediaTime()
    var dim = SIMD2<UInt32>(1, 1)

    init?(device: MTLDevice, metalSource: String) {
        guard let q = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = q
        super.init()

        guard device.supportsRaytracing else {
            fatalError("This GPU does not support Metal ray tracing.")
        }
        do {
            let opts = MTLCompileOptions()
            if #available(macOS 13.0, *) { opts.languageVersion = .version3_0 }
            let lib = try device.makeLibrary(source: metalSource, options: opts)
            let fn = lib.makeFunction(name: "rtKernel")!
            pipeline = try device.makeComputePipelineState(function: fn)
        } catch { fatalError("Shader compile/pipeline failed: \(error)") }

        buildScene()
    }

    // ---- scene ------------------------------------------------------------
    func buildScene() {
        mesh.addFloor(20.0, 0.0)

        // centre: morphing supertoroid (clear glass)
        torusOffset = mesh.reserveGrid(U, Vr, SIMD3(1, 1, 1), 2.0)

        // six coloured glass shapes around the centre
        let ring: Float = 6.2
        func ringPos(_ k: Int) -> SIMD3<Float> {
            let ang = Float(k) / 6.0 * 2 * .pi
            return SIMD3(cosf(ang) * ring, 0, sinf(ang) * ring)
        }
        var p = ringPos(0); mesh.addSphere(SIMD3(p.x, 1.3, p.z), 1.3, SIMD3(0.55, 0.72, 1.00), 0, 2)
        p = ringPos(1); mesh.addDiamond(SIMD3(p.x, 1.5, p.z), 1.25, SIMD3(1.00, 0.65, 0.72), 2)
        p = ringPos(2); mesh.addCube(SIMD3(p.x, 1.0, p.z), 2.0, SIMD3(0.55, 0.95, 0.65), 2)
        p = ringPos(3); mesh.addPyramid(SIMD3(p.x, 0.0, p.z), 2.2, 2.4, SIMD3(1.00, 0.85, 0.45), 2)
        p = ringPos(4); mesh.addCylinder(SIMD3(p.x, 0.0, p.z), 1.05, 2.4, SIMD3(0.45, 0.92, 0.92), 2)
        p = ringPos(5); mesh.addDodecahedron(SIMD3(p.x, 1.55, p.z), 1.55, SIMD3(0.78, 0.60, 1.00), 2)

        // emissive area light (radius must match lightPos.w below)
        mesh.addSphere(SIMD3(4, 7, -2), 1.2, SIMD3(1.0, 0.95, 0.85), 0, 3)

        updateTorus(0)

        vertexBuffer = device.makeBuffer(bytes: mesh.verts, length: mesh.verts.count * MemoryLayout<Vertex>.stride, options: .storageModeShared)
        indexBuffer  = device.makeBuffer(bytes: mesh.idx,   length: mesh.idx.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)
    }

    // Regenerate the supertoroid grid into mesh.verts[torusOffset...].
    func updateTorus(_ t: Float) {
        let n: Float = 1.6 + 1.1 * (0.5 + 0.5 * sinf(t * 0.22))   // round <-> square tube
        let twist: Float = 3.0                                   // INTEGER -> ring closes
        let a: Float = 2.2
        let center = SIMD3<Float>(0, 2.7, 0)
        let q = simd_quatf(angle: 0.5 * t, axis: normalize(SIMD3<Float>(1, 0.25, 0.35)))

        func pos(_ i: Int, _ j: Int) -> SIMD3<Float> {
            let u = 2 * Float.pi * Float((i % U + U) % U) / Float(U)
            let v = 2 * Float.pi * Float((j % Vr + Vr) % Vr) / Float(Vr)
            var cv = abs(cosf(v)); if cv < 1e-9 { cv = 1e-9 }
            var sv = abs(sinf(v)); if sv < 1e-9 { sv = 1e-9 }
            let R = powf(powf(cv, n) + powf(sv, n), -1.0 / n)
            let phi = twist * u + v, r = a + R * cosf(phi)
            return SIMD3(r * cosf(u), R * sinf(phi), r * sinf(u))
        }
        let stride = Vr + 1
        for i in 0...U {
            for j in 0...Vr {
                let p = pos(i, j)
                let du = pos(i + 1, j) - pos(i - 1, j)
                let dv = pos(i, j + 1) - pos(i, j - 1)
                var nn = cross(du, dv)
                let nl = length(nn)
                nn = nl > 1e-5 ? nn / nl : normalize(p)
                let wp = q.act(p) + center
                let wn = q.act(nn)
                var vv = mesh.verts[torusOffset + i * stride + j]
                vv.px = wp.x; vv.py = wp.y; vv.pz = wp.z
                vv.nx = wn.x; vv.ny = wn.y; vv.nz = wn.z
                mesh.verts[torusOffset + i * stride + j] = vv
            }
        }
    }

    // ---- acceleration structures ------------------------------------------
    func descriptors() {
        triGeo.vertexBuffer = vertexBuffer
        triGeo.vertexBufferOffset = 0
        triGeo.vertexStride = MemoryLayout<Vertex>.stride
        triGeo.vertexFormat = .float3
        triGeo.indexBuffer = indexBuffer
        triGeo.indexBufferOffset = 0
        triGeo.indexType = .uint32
        triGeo.triangleCount = mesh.idx.count / 3
        primDesc.geometryDescriptors = [triGeo]
        primDesc.usage = .refit

        instDesc.instancedAccelerationStructures = [blas].compactMap { $0 }
        instDesc.instanceCount = 1
        instDesc.instanceDescriptorBuffer = instanceBuffer
    }

    func makeInstanceBuffer() {
        var inst = MTLAccelerationStructureInstanceDescriptor()
        var m = MTLPackedFloat4x3()
        m.columns.0 = MTLPackedFloat3Make(1, 0, 0)
        m.columns.1 = MTLPackedFloat3Make(0, 1, 0)
        m.columns.2 = MTLPackedFloat3Make(0, 0, 1)
        m.columns.3 = MTLPackedFloat3Make(0, 0, 0)
        inst.transformationMatrix = m
        inst.options = .opaque
        inst.mask = 0xFF
        inst.intersectionFunctionTableOffset = 0
        inst.accelerationStructureIndex = 0
        instanceBuffer = device.makeBuffer(bytes: &inst,
            length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride, options: .storageModeShared)
    }

    func buildAccel(_ cb: MTLCommandBuffer, firstBuild: Bool) {
        if firstBuild {
            makeInstanceBuffer()
            descriptors()
            let bs = device.accelerationStructureSizes(descriptor: primDesc)
            blas = device.makeAccelerationStructure(size: bs.accelerationStructureSize)
            blasScratch = device.makeBuffer(length: max(bs.buildScratchBufferSize, bs.refitScratchBufferSize),
                                            options: .storageModePrivate)
            descriptors() // refresh instDesc now that blas exists
            let ts = device.accelerationStructureSizes(descriptor: instDesc)
            tlas = device.makeAccelerationStructure(size: ts.accelerationStructureSize)
            tlasScratch = device.makeBuffer(length: max(ts.buildScratchBufferSize, ts.refitScratchBufferSize),
                                            options: .storageModePrivate)
        }
        let enc = cb.makeAccelerationStructureCommandEncoder()!
        if firstBuild {
            enc.build(accelerationStructure: blas!, descriptor: primDesc, scratchBuffer: blasScratch!, scratchBufferOffset: 0)
        } else {
            enc.refit(sourceAccelerationStructure: blas!, descriptor: primDesc,
                      destinationAccelerationStructure: blas!, scratchBuffer: blasScratch!, scratchBufferOffset: 0)
        }
        enc.build(accelerationStructure: tlas!, descriptor: instDesc, scratchBuffer: tlasScratch!, scratchBufferOffset: 0)
        enc.endEncoding()
    }

    // ---- textures ---------------------------------------------------------
    func makeAccumTextures(_ w: Int, _ h: Int) {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: max(w,1), height: max(h,1), mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        accumTex = device.makeTexture(descriptor: d)
        histTex  = device.makeTexture(descriptor: d)
        needsReset = true
    }

    // ---- camera helpers ---------------------------------------------------
    func eye() -> SIMD3<Float> {
        SIMD3(cosf(elevation) * cosf(azimuth), sinf(elevation), cosf(elevation) * sinf(azimuth)) * distance + target
    }
    func markMoved() { needsReset = true }

    // ---- MTKViewDelegate --------------------------------------------------
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        dim = SIMD2(UInt32(size.width), UInt32(size.height))
        makeAccumTextures(Int(size.width), Int(size.height))
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let accum = accumTex, let hist = histTex,
              let cb = queue.makeCommandBuffer() else { return }

        let t = Float(CACurrentMediaTime() - startTime)
        updateTorus(t)
        // upload the changed vertices (torus slice; simplest: whole buffer)
        memcpy(vertexBuffer.contents(), mesh.verts, mesh.verts.count * MemoryLayout<Vertex>.stride)

        let first = (blas == nil)
        buildAccel(cb, firstBuild: first)

        frameAccum = needsReset ? 0 : frameAccum &+ 1
        frameCounter = frameCounter &+ 1
        needsReset = false

        var U = Uniforms()
        let proj = perspective(Float.pi / 3, Float(dim.x) / Float(max(dim.y, 1)), 0.05, 200)
        let view4 = lookAt(eye(), target, SIMD3(0, 1, 0))
        U.viewInverse = view4.inverse
        U.projInverse = proj.inverse
        U.lightPos = SIMD4(4, 7, -2, 1.2)
        U.params   = SIMD4(t, 8, 1.0, 12)        // time, maxBounces, intensity, spp
        U.frame    = SIMD2(frameAccum, frameCounter)
        U.dim      = dim

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.setAccelerationStructure(tlas, bufferIndex: 1)
        enc.setBuffer(vertexBuffer, offset: 0, index: 2)
        enc.setBuffer(indexBuffer, offset: 0, index: 3)
        enc.setTexture(accum, index: 0)
        enc.setTexture(hist, index: 1)
        enc.setTexture(drawable.texture, index: 2)
        if let b = blas { enc.useResource(b, usage: .read) }     // referenced by the TLAS

        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(width: (Int(dim.x) + 7) / 8, height: (Int(dim.y) + 7) / 8, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cb.present(drawable)
        cb.commit()
    }
}

// ============================================================
// MARK: - View with input handling
// ============================================================

final class RTView: MTKView {
    weak var renderer: Renderer?
    private var last = NSPoint.zero

    override var acceptsFirstResponder: Bool { true }
    override func mouseDown(with e: NSEvent) { last = e.locationInWindow }
    override func mouseDragged(with e: NSEvent) {
        let p = e.locationInWindow
        let dx = Float(p.x - last.x), dy = Float(p.y - last.y); last = p
        guard let r = renderer else { return }
        r.azimuth -= dx * 0.01
        r.elevation = max(-1.4, min(1.4, r.elevation + dy * 0.01))
        r.markMoved()
    }
    override func scrollWheel(with e: NSEvent) {
        guard let r = renderer else { return }
        r.distance = max(4, min(60, r.distance - Float(e.scrollingDeltaY) * 0.03))
        r.markMoved()
    }
    override func keyDown(with e: NSEvent) {
        guard let r = renderer else { return }
        switch e.keyCode {
        case 53: NSApp.terminate(nil)                              // ESC
        case 15: r.azimuth = 0.9; r.elevation = 0.35; r.distance = 16; r.markMoved()  // R
        case 3:  window?.toggleFullScreen(nil)                     // F
        default: break
        }
    }
}

// ============================================================
// MARK: - App bootstrap
// ============================================================

func loadShaderSource() -> String {
    // Look for Shaders.metal next to the executable, then in the CWD.
    let exeDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
    for path in ["\(exeDir)/Shaders.metal", "Shaders.metal"] {
        if let s = try? String(contentsOfFile: path, encoding: .utf8) { return s }
    }
    fatalError("Shaders.metal not found next to the executable or in the working directory.")
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
}
let appDelegate = AppDelegate()
app.delegate = appDelegate

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device.") }
guard let renderer = Renderer(device: device, metalSource: loadShaderSource()) else { fatalError("Renderer init failed.") }

let frame = NSRect(x: 0, y: 0, width: 1000, height: 1000)
let window = NSWindow(contentRect: frame,
                      styleMask: [.titled, .closable, .miniaturizable, .resizable],
                      backing: .buffered, defer: false)
window.title = "Metal Hardware Ray Tracing"
window.center()

let view = RTView(frame: frame, device: device)
view.renderer = renderer
view.delegate = renderer
view.colorPixelFormat = .bgra8Unorm
view.framebufferOnly = false           // allow the compute kernel to write the drawable
view.preferredFramesPerSecond = 60
renderer.mtkView(view, drawableSizeWillChange: view.drawableSize)

window.contentView = view
window.makeKeyAndOrderFront(nil)
window.makeFirstResponder(view)
app.activate(ignoringOtherApps: true)
app.run()
