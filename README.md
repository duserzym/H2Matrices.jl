<img src="docs/src/assets/logo.svg" alt="H2Matrices.jl logo" width="96" align="right">

# H2Matrices.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21748179.svg)](https://doi.org/10.5281/zenodo.21748179)

_A package for assembling and factoring H²-matrices (hierarchical matrices with nested bases)._

So far this package is completely vibed out with AI agents, by feeding the HMatrices.jl library and the H2Lib C library as references. Please feel free to test and use with caution.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/duserzym/H2Matrices.jl", rev="v0.2.0")
```

## Overview

This package provides functionality for assembling, compressing, and performing
linear algebra with
[H²-matrices](https://en.wikipedia.org/wiki/Hierarchical_matrix) — hierarchical
matrices that exploit **shared nested bases** to achieve `O(N)` storage and
matrix–vector product cost.  It builds on
[HMatrices.jl](https://github.com/WaveProp/HMatrices.jl) for cluster trees,
kernel matrices, and ACA.

The accuracy and performance work is documented in detail on the
website: [algorithmic causes and changes](https://duserzym.github.io/H2Matrices.jl/stable/advances/),
[practical configuration](https://duserzym.github.io/H2Matrices.jl/stable/accuracy_performance/),
and [real-grain validation with downloadable data](https://duserzym.github.io/H2Matrices.jl/stable/validation/).
The explanation covers inherited interactions, ACA factor weighting, stored
adjoints, rank caps, implicit saturated bases, packet products, pass-through
bases, mixed-precision storage, absolute error control and safe concurrency.

### What changed in 0.2.0

- **Construction**: the H → H² conversion condenses ancestor interactions
  exactly, runs as Julia tasks and releases the intermediate H-matrix while it
  converts. On four real boundary-element grains (6,028-30,321 boundary nodes)
  the conversion was 5.7-9 times faster than v0.1.3 and the build's peak live
  heap roughly halved, for the same operator up to rounding.
- **Products**: a new packet engine runs every phase in parallel with explicit
  write ownership (bitwise independent of the worker count) and applies up to
  16 right-hand sides per pass; four-worker products were 1.6-1.9 times faster
  than v0.1.3's packet plan.
- **Storage**: exact pass-through bases (`passthrough=true`, 4-8% smaller),
  optional coupling truncation with Float32/Float16 tiers, and
  `H2MixedPacketMatvecPlan`, which stores low-weight rows in Float32 under a
  rigorous a priori bound (`precision_rtol`).
- **Error control**: opt-in absolute tolerances
  (`error_control=:global`, `atol`, `aca_atol`, `estimate_operator_scale`).
- **Breaking**: the conversion stores a different but equivalent
  representation (identity bases where nothing is truncated), conversion
  threads are on by default with several Julia threads and one BLAS thread,
  and `H2PacketMatvecPlan` is now parametric. See [CHANGELOG.md](CHANGELOG.md).

An application finding motivates new recommended settings: in Merrill's
end-to-end micromagnetic check, the tolerances validated on the PLAG066 grain
with v0.1.x (`rtol=1e-10`, block-relative) gave relative product errors near
1e-10 but missed a 1e-9 T tangent-torque gate on larger grains (up to 1.4e-8 T
on PLAG012). Absolute error control at `rtol=3e-12` with the mixed-precision
plan met the gate on all four grains (at most 1.6e-10 T) while storing 9-19%
less than the previously validated operator; builds on the three larger grains
were 2.9-3.7 times slower than with the previous settings. These are maxima
over sampled physical states, not bounds: states constructed to maximize the
error break the energy gate on PLAG012. See
[Large operators with tight absolute accuracy](#large-operators-with-tight-absolute-accuracy)
below.

For the purpose of illustration, let us consider an abstract matrix `K` with
entry `i,j` given by the evaluation of some _kernel function_ `G` on points
`X[i]` and `Y[j]`, where `X` and `Y` are vectors of points (in 3D here); that
is, `K[i,j] = G(X[i], Y[j])`.  This object can be constructed as follows:

```julia
using H2Matrices
using HMatrices: KernelMatrix, ClusterTree, GeometricSplitter, assemble_hmatrix
using StaticArrays, LinearAlgebra
using Plots

const Point3D = SVector{3,Float64}

# sample some points on a sphere
m = 50_000
X = Y = [Point3D(sin(θ)cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
         for (θ,ϕ) in zip(π*rand(m), 2π*rand(m))]

function G(x, y)
    d = norm(x - y) + 1e-8
    1 / (4π * d)
end

K = KernelMatrix(G, X, Y)
```

where we took `G` to be the free-space Green's function of Laplace's equation
in 3D (to avoid division-by-zero we added `1e-8` to the distance between
points).

The object `K` corresponds to a dense matrix, so converting it to a matrix can
be costly both in terms of memory and flops.  Instead, we can construct an
approximation to `K` as a hierarchical matrix using
[HMatrices.jl](https://github.com/WaveProp/HMatrices.jl), and then build an
H²-matrix from the same kernel:

```julia
# H-matrix (for comparison)
H = assemble_hmatrix(K; atol=1e-6)

# H²-matrix via Chebyshev interpolation
h2 = assemble_h2matrix(K; order=4)
```

We can now use `H` and `h2` as approximations to `K` for matrix–vector
products:

```julia
x = rand(m)
y_h  = H * x
y_h2 = h2 * x
```

For small problems you can compare against a dense reference. For large problems,
sample a few entries or matvecs:

```julia
sample = 42
exact_sample = sum(K[sample,j] * x[j] for j in 1:m)

println("H-matrix  sampled entry error: ", abs(y_h[sample]  - exact_sample))
println("H²-matrix sampled entry error: ", abs(y_h2[sample] - exact_sample))
println("H² summary: ", H2Matrices.compression_summary(h2))
```

```
H-matrix  sampled entry error: ≈ 2e-7
H²-matrix sampled entry error: ≈ 5e-4
```

> **Tip**: You can visualize the underlying block structure using Plots.jl
> recipes included in this package:
>
> ```julia
> plot(
>     plot(H;  title="H-matrix"),
>     plot(h2; title="H²-matrix");
>     layout=(1,2), size=(1000, 450)
> )
> ```
>
> which produces:
>
> ![H vs H² block structures](docs/src/assets/h_and_h2_matrix_block_structures.png)
>
> Admissible blocks are colored by rank: darker blue means lower rank, brighter
> gold means higher rank. Dense near-field blocks are shown in orange.

## Key Features

- **Chebyshev interpolation assembly** — build an H²-matrix directly from a
  kernel function using tensor-product Chebyshev interpolation.
- **Adaptive ACA-based assembly** — construct an H-matrix with ACA, then
  convert to H² format with shared nested bases.
- **O(N) matrix–vector product** — three-phase algorithm (forward transform →
  coupling → backward transform + near-field).
- **Recompression** — reduce basis ranks of an existing H²-matrix while
  controlling approximation error.
- **Diagnostics** — summarize storage, block counts, ranks, and sampled
  approximation errors.
- **Solver wrappers** — CG and restarted GMRES operate directly on compressed
  H² matrices through `mul!`.
- **Built on [HMatrices.jl](https://github.com/WaveProp/HMatrices.jl)** —
  leverages its cluster trees, admissibility conditions, and ACA.
- **Appropriate application domains** — ideal for large dense matrices arising from non-local
  operators, e.g. integral equations, kernel methods, covariance matrices. Based on my testing, it is most effective for matrices of size `N > 10_000` (depending on the kernel and desired accuracy).


## Accuracy and adjoints

Adaptive conversion preserves direct and inherited far-field interactions in
partner-QR-weighted nested bases. Unused parent bases have rank zero. Use
`strict=true` to reject a rank cap that prevents the requested local SVD
accuracy, and control initial ACA accuracy separately with `aca_rtol`:

```julia
h2 = assemble_h2matrix_adaptive(K; rtol=1e-9, aca_rtol=1e-10,
                                maxrank=120, strict=true)
y = h2 * x
z = adjoint(h2) * y
```

`adjoint(h2)` and `transpose(h2)` apply the exact transpose of the stored real
approximation. They reuse the existing bases and numerical blocks, including
independent row and column permutations and rectangular matrices.

The basis tolerance is local; it does not certify the total operator, field,
or solver error. Validate against a trusted reference at the accuracy required
by the application. Adaptive assembly still builds an H-matrix first, so peak
construction memory is larger than final H2 storage.

## Large operators with tight absolute accuracy

```julia
using H2Matrices, LinearAlgebra
import HMatrices
BLAS.set_num_threads(1)            # Julia threads drive conversion and products

h2 = assemble_h2matrix_adaptive(K; nmax=32,
    error_control=:global,         # ACA and basis tolerances aca_rtol*s and rtol*s,
    rtol=3e-12, aca_rtol=3e-13,    # s = estimate_operator_scale(...) (RMS row norm)
    maxrank=typemax(Int), strict=true,
    adm=HMatrices.StrongAdmissibilityStd(1.5),
    threads=true)                  # threaded H-matrix leaves; K must allow concurrent getblock!
plan = H2MixedPacketMatvecPlan(h2; workers=4, precision_rtol=1e-13,
                               passthrough=true, consume=true)  # h2 is unusable afterwards
mul!(y, plan, x)
mul!(g, adjoint(plan), z)          # exact transpose of the stored operator
mul!(Y, plan, X)                   # several right-hand sides
```

Merrill's default boundary operator uses these settings in its release built
on H2Matrices 0.2. They were selected against the dense operator on four
grains; they are not universal defaults, and the package's own defaults
remain block-relative (`error_control=:block`). On the larger grains the
mixed plan's multi-vector products were up to 25% slower per vector than
Float64 packets; `H2PacketMatvecPlan(h2; workers=4, passthrough=true,
consume=true)` stores Float64 packets (17-30% more bytes on these grains) when
batched products matter more than memory.

## Reusable matvec plans and recompression

```julia
plan = H2MatvecPlan(h2)
mul!(y, plan, x)
mul!(z, adjoint(plan), y)

recompress!(h2; rtol=1e-8, maxrank=512, strict=true)
plan = H2MatvecPlan(h2) # rebuild after changing the matrix
```

Plans flatten tree traversal, cache index permutations, and reuse coefficient
and vector buffers. Their numerical blocks are shared with `h2`, so they do
not duplicate the operator. Each concurrent worker needs its own plan; a
single plan must not be used concurrently. Neither source bases nor blocks
should be mutated while a plan is in use.

Strict recompression rejects insufficient rank caps and leaves the original
unchanged on failure. It uses a temporary copy, increasing peak setup memory.
It visits active descendants even when their parent has rank zero.

`H2LowRankMatvecPlan(h2; rtol=1e-10)` is an experimental matvec-only alternative
that stores selected coupling matrices as SVD factors when those factors use
less numeric storage. It retains bases and near-field blocks without retaining
the original operator or discarded couplings. `storage_bytes(plan)` reports
its numeric matrix storage, excluding scratch buffers and object overhead.
The coupling tolerance introduces another local approximation and needs
application-level validation; reduced storage does not guarantee faster
products. Its adjoint uses exactly the same stored factors. It also requires
one plan per worker and immutable shared numerical data.

## Compact, packet and mixed-precision plans

```julia
using LinearAlgebra
BLAS.set_num_threads(1)

compact = H2CompactMatvecPlan(h2)       # no extra truncation
plan = H2PacketMatvecPlan(h2; workers=4)
mul!(y, plan, x)
mul!(z, adjoint(plan), y)
worker_plan = copy(plan)               # shared matrices, independent scratch
```

The compact plan replaces saturated cluster bases with implicit identity
bases and moves their numerical action into couplings and parent transfers.
It works with nonorthogonal and overcomplete bases. This changes the stored
representation up to floating-point rounding, without relaxing tolerances.
The packet plan groups couplings into row packets and near-field blocks into
column packets, applied by fused long-column kernels. Every phase (upward
pass, interactions, slot reduction plus downward pass) runs in parallel with
explicit write ownership, so products are deterministic and bitwise
independent of the worker count. `mul!(Y, plan, X)` and its adjoint apply
several right-hand sides while streaming the operator once per block of up to
16 vectors.

`workers=1` is the default and supports allocation-free warmed products on
recent Julia compilers. Threaded plans allocate small task-scheduling objects.
Always use one plan per concurrent caller: `copy(plan)` shares numerical data
while copying all mutable scratch. Do not mutate shared numerical matrices
or plan metadata while workers are active. Plans are matvec-only operators.

`passthrough=true` folds weakly compressing transfer levels into couplings
(exact up to rounding). `H2MixedPacketMatvecPlan(h2; precision_rtol=1e-13)`
runs on the same packet engine and stores low-weight rotated coupling rows in
Float32 under the a priori bound `‖Ã - A‖_F ≤ precision_rtol · η`;
`precision_summary(plan)` reports the selection. `coupling_rtol`,
`coupling_scale` and `coupling_precision` are further approximations that
need their own validation.

The v0.1.x validation: on the real PLAG066 campaign grain (19,901 nodes,
100,602 tetrahedra, 6,028 boundary nodes), the four-worker packet plan retained the original basis
and ACA tolerances (`1e-10` and `1e-11`) while using 302.75 MB of numeric
storage versus 337.78 MB for H2 and 378.59 MB for H. NEB polishing took
52.2 s versus 53.9 s for H in paired single runs. Newly generated LEM/NEB
states differed from H by at most 5.5e-11 T in tangent torque. These results
are specific to this grain and path, not a general performance guarantee.
Construction still uses a temporary H matrix; final storage is not peak RSS.
The same tolerances missed the 1e-9 T torque gate on larger grains (see
above); the 0.2.0 four-grain results are on the
[validation page](https://duserzym.github.io/H2Matrices.jl/stable/validation/).

For the unregistered package, install a reproducible release with:

```julia
using Pkg
Pkg.add(url="https://github.com/duserzym/H2Matrices.jl", rev="v0.2.0")
```

## Documentation

For more information, see the
[documentation](docs/src/index.md) and the
[examples](docs/src/examples.md).

The current development plan is tracked in [ROADMAP.md](ROADMAP.md).

## References

- Hackbusch, Wolfgang. *Hierarchical matrices: algorithms and analysis.* Vol. 49. Heidelberg: Springer, 2015.
- Börm, Steffen. *Efficient numerical methods for non-local operators.* EMS, 2010.
- Hackbusch, W., Khoromskij, B., & Sauter, S. A. (2000). On H2-Matrices. In H.-J. Bungartz, R. H. W. Hoppe, & C. Zenger (Eds.), Lectures on Applied Mathematics (pp. 9–29). Springer. https://doi.org/10.1007/978-3-642-59709-1_2
- HMatrices.jl: https://github.com/IntegralEquations/HMatrices.jl
- H2Lib: https://github.com/H2Lib/H2Lib

Julia 1.13 or later is required. CI tracks the latest stable Julia release.
