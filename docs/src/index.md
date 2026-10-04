# H2Matrices.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21748179.svg)](https://doi.org/10.5281/zenodo.21748179)

*Fast hierarchical matrix algebra with nested bases in Julia.*

## Accuracy and performance advances

v0.1.x corrected nested-basis conversion and stored adjoints, then improved
retained memory and repeated-product throughput using implicit saturated bases,
reusable scratch and contiguous interaction packets. On a real
6,028-boundary-node PLAG066 grain, the selected four-worker plan used 302.75 MB
versus 337.78 MB for corrected H² and 378.59 MB for H, while retaining the
original ACA/basis tolerances.

**0.2.0** makes construction and products faster and adds storage and
error-control options:

- The H → H² conversion condenses ancestor interactions exactly, runs as Julia
  tasks and releases the intermediate H-matrix while converting: 5.7-9 times
  faster than v0.1.3 on four grains of 6,028-30,321 boundary nodes, for the
  same operator up to rounding.
- A new packet engine runs every product phase in parallel with explicit write
  ownership (bitwise independent of the worker count) and applies several
  right-hand sides per pass; four-worker products were 1.6-1.9 times faster.
- Exact pass-through bases, optional coupling truncation with reduced-precision
  tiers, and `H2MixedPacketMatvecPlan`, which stores low-weight rows in
  Float32 under a rigorous a priori bound.
- Opt-in absolute error control (`error_control=:global`).

An application check found that the tolerances validated on PLAG066 are not
enough on larger grains: their relative product errors stayed near 1e-10, but
the tangent torque of Merrill's micromagnetic energy missed a 1e-9 T gate by up
to 14 times on the 30,321-node PLAG012 grain. Absolute error control at a
tighter tolerance with the mixed-precision plan met the gate on all four grains
with 9-19% less storage than the previous operator, at 2.9-3.7 times the build
time. The [practical guide](accuracy_performance.md#Recommended-settings-for-tight-absolute-accuracy)
gives these settings.

Read [how the advances work](advances.md), follow the
[practical accuracy/performance guide](accuracy_performance.md), or inspect the
[validation protocol and downloadable results](validation.md). The measurements
cover four grains of one application on one machine; retained storage is not
peak assembly memory.

## What is an H²-matrix?

Many problems in physics and engineering lead to large dense matrices that arise
from evaluating a kernel function between pairs of points — for example, the
Green's function in electrostatics, gravity in geophysics, or the demagnetizing
field in micromagnetics.  These matrices are often too large to store or multiply
directly (``O(N^2)`` cost), yet they contain a great deal of structure that can
be exploited.

**H²-matrices** (hierarchical matrices with nested bases) can reduce such kernel
matrices to linear storage and matrix–vector product cost when ranks and
interaction counts remain controlled.
They do this by:

1. Recursively partitioning the row and column index sets into a **cluster tree**.
2. Approximating well-separated (far-field) blocks with a low-rank factorization
   ``A_{τσ} ≈ V_τ \, S_{τσ} \, W_σ^{\!\top}``.
3. **Sharing** the basis matrices ``V_τ`` and ``W_σ`` across blocks — this is
   what distinguishes H²-matrices from ordinary H-matrices and can yield linear
   complexity when ranks and interaction counts remain controlled.

## Key Features

- **Chebyshev interpolation assembly** — build an H²-matrix directly from a
  kernel function using tensor-product Chebyshev interpolation.
- **Adaptive ACA-based assembly** — construct an H-matrix with ACA,
  then convert to H² format with shared nested bases.
- **O(N) matrix–vector product** — three-phase algorithm (forward transform →
  coupling interaction → backward transform + near-field).
- **Recompression** — reduce basis ranks of an existing H²-matrix while
  controlling approximation error (weight-based SVD truncation).
- **Built on [HMatrices.jl](https://github.com/WaveProp/HMatrices.jl)** —
  leverages its cluster trees, admissibility conditions, and ACA implementation.

## Quick Start

```julia
using H2Matrices
using HMatrices: KernelMatrix, ClusterTree, GeometricSplitter
using StaticArrays, LinearAlgebra

# Define point sets (e.g., source and target points)
N = 1000
src = [SVector{2,Float64}(rand(), rand()) for _ in 1:N]
tgt = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:N]

# Define a 1/r potential kernel on planar point sets
K = KernelMatrix(src, tgt) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end

# Assemble an H²-matrix using Chebyshev interpolation
h2 = assemble_h2matrix(K; order=4)

# Matrix–vector product (O(N) cost)
x = randn(N)
y = h2 * x
```

See [Getting Started](@ref) for a more detailed walkthrough, or jump to
[Examples](@ref) for complete worked problems.

## Who is this for?

This package is designed for researchers and students in:

- **Micromagnetics** — fast evaluation of demagnetizing field kernels.
- **Geophysics** — gravitational and magnetic potential field modeling.
- **Electrostatics / BEM** — boundary element methods with Laplace or Helmholtz
  kernels.
- **Any field** where you need to multiply large dense kernel matrices fast.

For compressible kernels with controlled ranks and interaction counts, nested
bases can support linear storage and product complexity. Actual ranks, near-field
storage and setup memory should be measured for the intended application.
