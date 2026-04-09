# Getting Started

## Installation

H2Matrices.jl depends on [HMatrices.jl](https://github.com/WaveProp/HMatrices.jl)
for cluster trees, kernel matrices, and ACA.  Install both with:

```julia
using Pkg
Pkg.add(url="https://github.com/WaveProp/HMatrices.jl")
Pkg.add(url="https://github.com/yimingzhang/H2Matrices.jl")
```

## Basic Workflow

Every H²-matrix computation follows the same pattern:

1. **Define your points** — source and target locations.
2. **Define your kernel** — the function ``G(x, y)`` to compress.
3. **Build cluster trees** — hierarchical partitioning of the points.
4. **Assemble the H²-matrix** — using Chebyshev interpolation or adaptive ACA.
5. **Use it** — matrix–vector products, dense conversion, etc.

### Step 1: Define Points

Points are represented as `SVector`s from
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl):

```julia
using StaticArrays

N = 500
src = [SVector{2,Float64}(rand(), rand()) for _ in 1:N]
tgt = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:N]
```

The source and target sets should be **spatially separated** so that most
block pairs satisfy the admissibility condition.  (If they overlap, the method
still works but most blocks will be near-field/dense.)

### Step 2: Define a Kernel

Wrap your kernel function in an `HMatrices.KernelMatrix`:

```julia
using HMatrices: KernelMatrix
using LinearAlgebra

K = KernelMatrix(src, tgt) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end
```

This represents the ``N \times N`` matrix ``K_{ij} = G(\text{src}_i, \text{tgt}_j)``
without forming it explicitly.

### Step 3: Build Cluster Trees

Cluster trees recursively partition the point sets:

```julia
using HMatrices: ClusterTree, GeometricSplitter

Xclt = ClusterTree(deepcopy(src), GeometricSplitter(; nmax=30))
Yclt = ClusterTree(deepcopy(tgt), GeometricSplitter(; nmax=30))
```

The `nmax` parameter controls the leaf size (maximum points per leaf cluster).
Smaller values create deeper trees with more levels.

!!! tip
    Use `deepcopy` when passing points to `ClusterTree` — the constructor
    reorders the points in-place.

### Step 4: Assemble the H²-Matrix

**Option A: Chebyshev interpolation** (deterministic, good for smooth kernels):

```julia
using H2Matrices

h2 = assemble_h2matrix(K, Xclt, Yclt; order=4, global_index=true)
```

The `order` parameter controls the Chebyshev interpolation order per dimension.
Higher order gives better accuracy but higher rank (``k = \text{order}^d``).

**Option B: Adaptive ACA → H²** (data-driven, adapts to kernel smoothness):

```julia
h2 = assemble_h2matrix_adaptive(K, Xclt, Yclt; rtol=1e-6, maxrank=50)
```

This first builds an H-matrix via ACA, then converts to H² format with shared
nested bases.  The `rtol` parameter controls the approximation tolerance.

### Step 5: Use the H²-Matrix

**Matrix–vector product** (uses the O(N) three-phase algorithm):

```julia
x = randn(N)
y = h2 * x          # operator syntax
```

Or with explicit in-place multiplication:

```julia
y = zeros(N)
mul!(y, h2, x)                # y = h2 * x
mul!(y, h2, x, 2.0, 1.0)     # y = y + 2 * h2 * x
```

**Convert to dense** (for debugging or comparison):

```julia
M = Matrix(h2)  # N × N dense matrix
```

**Check compression**:

```julia
println("Size: ", size(h2))
println("Compression ratio: ", H2Matrices.compression_ratio(h2))
```

## Recompression

If your H²-matrix has higher ranks than needed, you can recompress it:

```julia
recompress!(h2; rtol=1e-4, maxrank=30)
```

This modifies the matrix in-place, reducing basis ranks while keeping the
approximation error within the specified tolerance.

## Choosing Parameters

| Parameter | Meaning | Typical values |
|:----------|:--------|:---------------|
| `order`   | Chebyshev interpolation order per dimension | 3–6 |
| `rtol`    | Relative truncation tolerance | `1e-4` to `1e-8` |
| `maxrank` | Maximum rank per cluster | 30–100 |
| `nmax`    | Max points per leaf cluster | 20–64 |
| `η` (via `StrongAdmissibilityStd`) | Admissibility parameter | 2–3 |

**Rules of thumb:**
- For 2D problems, `order=4` gives rank ``k = 16`` per cluster.
- For 3D problems, `order=3` gives rank ``k = 27`` per cluster.
- Adaptive assembly often achieves lower ranks than Chebyshev for the same accuracy.
- Recompression after Chebyshev assembly can significantly reduce ranks.
