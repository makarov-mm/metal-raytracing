#!/bin/sh
# Build the Metal ray tracer (single Swift file; the .metal shader is compiled
# at runtime from Shaders.metal, which must stay next to the executable).
set -e

swiftc main.swift -O -o RayTracer \
    -framework Cocoa -framework Metal -framework MetalKit

echo "Built ./RayTracer"
echo "Run with:  ./RayTracer        (keep Shaders.metal in the same folder)"