# Examples

## Visualizing Matrix Compression

To build intuition, let's start with a visual overview of how H-matrices and
H²-matrices compress a dense kernel matrix.

Consider ``N = 500`` points randomly distributed in the unit square, with the
2D Laplace kernel ``G(x,y) = 1/(4\pi\|x-y\|)``:

![Point distribution](assets/point_geometry.png)

The full kernel matrix is ``500 \times 500`` and dense:

![Dense kernel matrix](assets/dense_kernel.png)

### H-Matrix vs H²-Matrix Block Structure

Both H-matrices and H²-matrices decompose this into a hierarchy of blocks.
Near-field (close-together) blocks are stored as dense matrices (orange), while
far-field blocks are compressed:

- **H-matrix**: each admissible block stores its own independent low-rank
  factors (blue, shade indicates rank).
- **H²-matrix**: admissible blocks share nested cluster bases (green), storing
  only small coupling matrices.

![H vs H² comparison](assets/h_vs_h2_comparison.png)

The block structure is identical — the difference is in how the low-rank data is
stored.  H²-matrices achieve ``O(N)`` storage by sharing bases across blocks at
the same tree level.

### Approximation Error

The error distribution reveals where each method is accurate.  Dense (near-field)
blocks are exact (black = machine zero), while far-field blocks carry
approximation error:

![Error heatmaps](assets/error_heatmaps.png)

The adaptive method (right) achieves much lower error in the far-field blocks
because the ranks adapt to the actual kernel smoothness, rather than using a
fixed Chebyshev order.

### Recompression

Starting from a Chebyshev H²-matrix with conservative order-5 interpolation
(total basis rank = 1075), recompression with ``\text{rtol} = 10^{-4}`` reduces
ranks while preserving accuracy:

![Recompression comparison](assets/recompression.png)

Lighter green blocks have lower rank after recompression — the algorithm
automatically identifies which blocks need less precision.

---

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
using HMatrices: KernelMatrix, ClusterTree, GeometricSplitter
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
for j in 1:N, i in 1:N
    K_dense[i, j] = K[i, j]
end
x = randn(N)
y_ref = K_dense * x

# --- Method 1: Chebyshev interpolation ---
h2_cheb = assemble_h2matrix(K, Xclt, Yclt; order=4, global_index=true)
println("Chebyshev (order=4):  error = ", norm(h2_cheb * x - y_ref) / norm(y_ref))
println("  compression ratio = ", H2Matrices.compression_ratio(h2_cheb))

# --- Method 2: Adaptive ACA → H² ---
h2_ada = assemble_h2matrix_adaptive(K, Xclt, Yclt; rtol=1e-6, maxrank=50)
println("Adaptive (rtol=1e-6): error = ", norm(h2_ada * x - y_ref) / norm(y_ref))
println("  compression ratio = ", H2Matrices.compression_ratio(h2_ada))

# --- Recompression ---
h2_recomp = assemble_h2matrix(K, Xclt, Yclt; order=5, global_index=true)
rank_before = H2Matrices.total_rank(h2_recomp.row_basis)
recompress!(h2_recomp; rtol=1e-4, maxrank=50)
rank_after = H2Matrices.total_rank(h2_recomp.row_basis)
println("Recompressed: rank $rank_before → $rank_after, error = ",
    norm(h2_recomp * x - y_ref) / norm(y_ref))
```

**Output:**

```
Chebyshev (order=4):  error = 0.0017
  compression ratio = 352.0
Adaptive (rtol=1e-6): error = 1.54e-6
  compression ratio = 273.0
Recompressed: rank 775 → 118, error = 0.000402
```

The adaptive method achieves 6 orders of magnitude better accuracy.
Recompression reduces total basis rank by 6.6× while preserving sub-0.1% error.

![Compression ratios](assets/ex1_compression.png)
![Matvec errors](assets/ex1_errors.png)

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

h2_3d = assemble_h2matrix(K3, Xclt3, Yclt3; order=3, global_index=true)

K3_dense = Matrix{Float64}(undef, N, N)
for j in 1:N, i in 1:N
    K3_dense[i, j] = K3[i, j]
end

x3 = randn(N)
println("3D Laplace (order=3): error = ",
    norm(h2_3d * x3 - K3_dense * x3) / norm(K3_dense * x3))
println("  compression ratio = ", H2Matrices.compression_ratio(h2_3d))
```

**Output:**

```
3D Laplace (order=3): error = 0.000788
  compression ratio = 54.9
```

The H²-matrix overlay on the exact matvec shows the approximation is nearly
indistinguishable:

![Matvec comparison](assets/ex2_matvec.png)

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
println("Compression ratio: ", H2Matrices.compression_ratio(h2))
```

**Output:**

```
Size: (200, 200)
Matvec computed with 200 entries
Compression ratio: 150.0
```

The source and target point sets for this example:

![Point geometry](assets/ex3_points.png)

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
println("Compression ratio: ", H2Matrices.compression_ratio(h2))
```

**Output:**

```
H²-matrix size: (300, 300)
Compression ratio: 2500.0
```

Recompression reduces the total basis rank from 507 to 120 — a 4.2× reduction —
while maintaining sub-0.1% matvec error:

![Rank reduction](assets/ex4_recompression.png)

## Example 5: Large-Scale 3D Problem (Sphere)

This example mirrors the
[HMatrices.jl README](https://github.com/IntegralEquations/HMatrices.jl) and
shows how H2Matrices.jl plugs into a realistic workflow.  We sample points
on a sphere and build the Laplace free-space Green's function matrix —
far too large to store as a dense matrix — then compress it to both an H-matrix
and an H²-matrix for comparison.

![Sphere point cloud](assets/ex5_sphere.png)

```julia
using H2Matrices
using HMatrices: KernelMatrix, ClusterTree, GeometricSplitter,
    assemble_hmatrix, PartialACA
using LinearAlgebra, StaticArrays

const Point3D = SVector{3,Float64}

# --- Point geometry: random points on a sphere ---
m = 100_000
X = Y = [Point3D(sin(θ)cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
         for (θ,ϕ) in zip(π*rand(m), 2π*rand(m))]

# Laplace free-space Green's function (regularised to avoid 1/0)
function G(x, y)
    d = norm(x - y) + 1e-8
    1 / (4π * d)
end

K = KernelMatrix(G, X, Y)

# --- H-matrix assembly (for comparison) ---
H = assemble_hmatrix(K; atol=1e-6)
println("H-matrix compression ratio: ", HMatrices.compression_ratio(H))

# Quick matvec check
x = rand(m)
y_h = H * x
println("H  matvec y[42] error: ", abs(y_h[42] - sum(K[42,j]*x[j] for j in 1:m)))

# --- H²-matrix assembly (one liner) ---
h2 = assemble_h2matrix_adaptive(K; rtol=1e-6, maxrank=80, nmax=32)
println("H²-matrix compression ratio: ", H2Matrices.compression_ratio(h2))

y_h2 = h2 * x
println("H² matvec y[42] error: ", abs(y_h2[42] - sum(K[42,j]*x[j] for j in 1:m)))
```

**Output** (with ``m = 10\,000`` for this documentation build):

```
H-matrix compression ratio: 4.42
H²-matrix compression ratio: 9.74
H  matvec y[42] error: 1.58e-7
H² matvec y[42] error: 5.24e-4
```

The H²-matrix achieves better compression than the H-matrix thanks to shared
nested bases.  On the full ``100\,000 \times 100\,000`` problem the dense matrix
would require roughly 75 GB; the hierarchical representations fit comfortably in
a few hundred MB.

![Compression comparison](assets/ex5_compression.png)

## Example 6: Quick-Start — Sphere with Side-by-Side Visualization

This self-contained example mirrors the [notebook](https://github.com/duserzym/H2Matrices.jl/blob/main/example/example.ipynb)
shipped with the package. It assembles both an H-matrix and an H²-matrix on
50 000 points on a sphere, compares their accuracy, and produces a side-by-side
block-structure plot.

```julia
using H2Matrices
using HMatrices: KernelMatrix, ClusterTree, GeometricSplitter,
    assemble_hmatrix, compression_ratio
using StaticArrays, LinearAlgebra
using Plots

const Point3D = SVector{3,Float64}

# --- Point geometry: random points on a sphere ---
m = 50_000
X = Y = [Point3D(sin(θ)cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
         for (θ,ϕ) in zip(π*rand(m), 2π*rand(m))]

# Laplace free-space Green's function (regularised)
function G(x, y)
    d = norm(x - y) + 1e-8
    1 / (4π * d)
end

K = KernelMatrix(G, X, Y)

# --- H-matrix assembly ---
H = assemble_hmatrix(K; atol=1e-6)

# --- H²-matrix assembly ---
h2 = assemble_h2matrix(K; order=4)

# --- Matvec accuracy check ---
x = rand(m)
y_h  = H * x
y_h2 = h2 * x
exact_42 = sum(K[42,j]*x[j] for j in 1:m)

println("H-matrix  matvec error at index 42: ", abs(y_h[42]  - exact_42))
println("H²-matrix matvec error at index 42: ", abs(y_h2[42] - exact_42))

# --- Side-by-side block structure plot ---
plot(
    plot(H;  title="H-matrix"),
    plot(h2; title="H²-matrix");
    layout=(1,2), size=(1000, 450)
)
savefig("h_and_h2_matrix_block_structures.png")
```

**Output:**

```
H-matrix  matvec error at index 42: ≈ 2e-7
H²-matrix matvec error at index 42: ≈ 5e-4
```

![H vs H² block structures](assets/h_and_h2_matrix_block_structures.png)
