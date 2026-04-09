# Examples

## Example 1: 2D Laplace Kernel

The Laplace Green's function in 2D (really the free-space fundamental solution
in the plane) is one of the most common kernels in potential theory:

```math
G(x, y) = \frac{1}{4\pi \|x - y\|}
```

This example compares Chebyshev and adaptive assembly, and demonstrates
recompression.

```julia
using H2Matrices
using HMatrices: KernelMatrix, ClusterTree, GeometricSplitter,
    assemble_hmatrix, PartialACA, loc2glob
using StaticArrays, LinearAlgebra, Random

Random.seed!(42)

# --- Setup ---
N = 300
src = [SVector{2,Float64}(rand(), rand()) for _ in 1:N]
tgt = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:N]

K = KernelMatrix(src, tgt) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end

Xclt = ClusterTree(deepcopy(src), GeometricSplitter(; nmax=30))
Yclt = ClusterTree(deepcopy(tgt), GeometricSplitter(; nmax=30))

# --- Dense reference (for error measurement) ---
K_dense = Matrix{Float64}(undef, N, N)
rp, cp = loc2glob(Xclt), loc2glob(Yclt)
for j in 1:N, i in 1:N
    K_dense[i, j] = K[rp[i], cp[j]]
end
x = randn(N)
y_ref = K_dense * x

# --- Method 1: Chebyshev interpolation ---
h2_cheb = assemble_h2matrix(K, Xclt, Yclt; order=4, global_index=true)
err_cheb = norm(h2_cheb * x - y_ref) / norm(y_ref)
println("Chebyshev (order=4):  error = $(round(err_cheb; sigdigits=3))")
println("  compression ratio = $(round(H2Matrices.compression_ratio(h2_cheb); sigdigits=3))")

# --- Method 2: Adaptive ACA → H² ---
h2_ada = assemble_h2matrix_adaptive(K, Xclt, Yclt; rtol=1e-6, maxrank=50)
err_ada = norm(h2_ada * x - y_ref) / norm(y_ref)
println("Adaptive (rtol=1e-6): error = $(round(err_ada; sigdigits=3))")
println("  compression ratio = $(round(H2Matrices.compression_ratio(h2_ada); sigdigits=3))")

# --- Recompression ---
h2_recomp = assemble_h2matrix(K, Xclt, Yclt; order=5, global_index=true)
rank_before = H2Matrices.total_rank(h2_recomp.row_basis)

recompress!(h2_recomp; rtol=1e-4, maxrank=50)
rank_after = H2Matrices.total_rank(h2_recomp.row_basis)
err_recomp = norm(h2_recomp * x - y_ref) / norm(y_ref)

println("Recompressed (order=5 → rtol=1e-4):")
println("  rank: $rank_before → $rank_after")
println("  error = $(round(err_recomp; sigdigits=3))")
```

## Example 2: 3D Laplace Kernel

The same workflow extends to three dimensions — common in geophysics (gravity
modeling) and micromagnetics (demagnetizing field):

```julia
Random.seed!(123)

N = 200
src3 = [SVector{3,Float64}(rand(), rand(), rand()) for _ in 1:N]
tgt3 = [SVector{3,Float64}(3.0 + rand(), rand(), rand()) for _ in 1:N]

K3 = KernelMatrix(src3, tgt3) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end

Xclt3 = ClusterTree(deepcopy(src3), GeometricSplitter(; nmax=20))
Yclt3 = ClusterTree(deepcopy(tgt3), GeometricSplitter(; nmax=20))

# Chebyshev assembly (order=3 → rank 27 per cluster in 3D)
h2_3d = assemble_h2matrix(K3, Xclt3, Yclt3; order=3, global_index=true)

# Dense reference
K3_dense = Matrix{Float64}(undef, N, N)
rp3, cp3 = loc2glob(Xclt3), loc2glob(Yclt3)
for j in 1:N, i in 1:N
    K3_dense[i, j] = K3[rp3[i], cp3[j]]
end

x3 = randn(N)
err_3d = norm(h2_3d * x3 - K3_dense * x3) / norm(K3_dense * x3)
println("3D Laplace (order=3): error = $(round(err_3d; sigdigits=3))")
println("  compression ratio = $(round(H2Matrices.compression_ratio(h2_3d); sigdigits=3))")
```

## Example 3: Adaptive Assembly with Automatic Tree Construction

For convenience, you can skip manual cluster tree construction:

```julia
Random.seed!(42)

N = 200
src = [SVector{2,Float64}(rand(), rand()) for _ in 1:N]
tgt = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:N]

K = KernelMatrix(src, tgt) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end

# One-liner: builds trees, runs ACA, converts to H²
h2 = assemble_h2matrix_adaptive(K; rtol=1e-6, maxrank=50, nmax=32)

println("Size: ", size(h2))
x = randn(N)
y = h2 * x
println("Matvec computed with $(length(y)) entries")
```

## Example 4: H-Matrix to H²-Matrix Conversion

If you already have an H-matrix from HMatrices.jl, you can convert it
directly:

```julia
using HMatrices: assemble_hmatrix, PartialACA, StrongAdmissibilityStd

Random.seed!(42)

N = 300
src = [SVector{2,Float64}(rand(), rand()) for _ in 1:N]
tgt = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:N]

K = KernelMatrix(src, tgt) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end

Xclt = ClusterTree(deepcopy(src), GeometricSplitter(; nmax=30))
Yclt = ClusterTree(deepcopy(tgt), GeometricSplitter(; nmax=30))

# Step 1: Build H-matrix with ACA
hmat = assemble_hmatrix(K, Xclt, Yclt;
    comp=PartialACA(; rtol=1e-10),
    global_index=true, threads=false)

# Step 2: Convert to H², gaining shared nested bases
h2 = compress_hmatrix_to_h2(hmat; rtol=1e-6, maxrank=50)

# Step 3: Optionally recompress
recompress!(h2; rtol=1e-4, maxrank=30)

println("H²-matrix size: ", size(h2))
println("Compression ratio: ", round(H2Matrices.compression_ratio(h2); sigdigits=3))
```
