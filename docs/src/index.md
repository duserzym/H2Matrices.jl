# H2Matrices.jl

*Fast hierarchical matrix algebra with nested bases in Julia.*

[![Build Status](https://github.com/yimingzhang/H2Matrices.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/yimingzhang/H2Matrices.jl/actions/workflows/CI.yml)

## What is an H²-matrix?

Many problems in physics and engineering lead to large dense matrices that arise
from evaluating a kernel function between pairs of points — for example, the
Green's function in electrostatics, gravity in geophysics, or the demagnetizing
field in micromagnetics.  These matrices are often too large to store or multiply
directly (``O(N^2)`` cost), yet they contain a great deal of structure that can
be exploited.

**H²-matrices** (hierarchical matrices with nested bases) compress such kernel
matrices down to ``O(N)`` storage and ``O(N)`` matrix–vector product cost.
They do this by:

1. Recursively partitioning the row and column index sets into a **cluster tree**.
2. Approximating well-separated (far-field) blocks with a low-rank factorization
   ``A_{τσ} ≈ V_τ \, S_{τσ} \, W_σ^{\!\top}``.
3. **Sharing** the basis matrices ``V_τ`` and ``W_σ`` across blocks — this is
   what distinguishes H²-matrices from ordinary H-matrices and yields ``O(N)``
   instead of ``O(N \log N)`` complexity.

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

# Define a kernel (2D Laplace Green's function)
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

If you can write down your kernel function ``G(x, y)``, this package can compress
and apply the resulting matrix in linear time.
