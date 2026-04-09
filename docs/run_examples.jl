#!/usr/bin/env julia
# Run all 5 examples, capture printed output, and generate per-example figures.
# Outputs are saved to docs/src/assets/

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.resolve()
Pkg.instantiate()

using H2Matrices
using HMatrices: KernelMatrix, ClusterTree, GeometricSplitter,
    assemble_hmatrix, PartialACA, StrongAdmissibilityStd
import HMatrices
using StaticArrays, LinearAlgebra, Random
using Plots

assets = joinpath(@__DIR__, "src", "assets")
mkpath(assets)

# =========================================================================
# Example 1: 2D Laplace Kernel
# =========================================================================
println("="^60)
println("EXAMPLE 1: 2D Laplace Kernel")
println("="^60)

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

K_dense = Matrix{Float64}(undef, N, N)
for j in 1:N, i in 1:N
    K_dense[i, j] = K[i, j]
end
x1 = randn(N)
y_ref = K_dense * x1

# Chebyshev
h2_cheb = assemble_h2matrix(K, Xclt, Yclt; order=4, global_index=true)
err_cheb = norm(h2_cheb * x1 - y_ref) / norm(y_ref)
cr_cheb = H2Matrices.compression_ratio(h2_cheb)
println("Chebyshev (order=4):  error = $(round(err_cheb; sigdigits=3))")
println("  compression ratio = $(round(cr_cheb; sigdigits=3))")

# Adaptive
h2_ada = assemble_h2matrix_adaptive(K, Xclt, Yclt; rtol=1e-6, maxrank=50)
err_ada = norm(h2_ada * x1 - y_ref) / norm(y_ref)
cr_ada = H2Matrices.compression_ratio(h2_ada)
println("Adaptive (rtol=1e-6): error = $(round(err_ada; sigdigits=3))")
println("  compression ratio = $(round(cr_ada; sigdigits=3))")

# Recompression
h2_recomp = assemble_h2matrix(K, Xclt, Yclt; order=5, global_index=true)
rank_before = H2Matrices.total_rank(h2_recomp.row_basis)
recompress!(h2_recomp; rtol=1e-4, maxrank=50)
rank_after = H2Matrices.total_rank(h2_recomp.row_basis)
err_recomp = norm(h2_recomp * x1 - y_ref) / norm(y_ref)
println("Recompressed (order=5 → rtol=1e-4):")
println("  rank: $rank_before → $rank_after")
println("  error = $(round(err_recomp; sigdigits=3))")

# --- Ex1 Plot: compression ratio and error bar chart ---
p1 = bar(["Chebyshev\n(order=4)", "Adaptive\n(rtol=1e-6)", "Recompressed\n(order=5→rtol=1e-4)"],
    [cr_cheb, cr_ada, H2Matrices.compression_ratio(h2_recomp)],
    ylabel="Compression Ratio", title="Example 1: Compression Ratios",
    legend=false, color=[:steelblue, :darkorange, :seagreen],
    bar_width=0.6, size=(600, 350))
savefig(p1, joinpath(assets, "ex1_compression.png"))
println("Saved ex1_compression.png")

p1e = bar(["Chebyshev\n(order=4)", "Adaptive\n(rtol=1e-6)", "Recompressed\n(order=5→rtol=1e-4)"],
    [err_cheb, err_ada, err_recomp],
    ylabel="Relative Error", title="Example 1: Matvec Errors",
    legend=false, color=[:steelblue, :darkorange, :seagreen], yscale=:log10,
    bar_width=0.6, size=(600, 350))
savefig(p1e, joinpath(assets, "ex1_errors.png"))
println("Saved ex1_errors.png")

# =========================================================================
# Example 2: 3D Laplace Kernel
# =========================================================================
println("\n" * "="^60)
println("EXAMPLE 2: 3D Laplace Kernel")
println("="^60)

Random.seed!(123)

N2 = 200
src3 = [SVector{3,Float64}(rand(), rand(), rand()) for _ in 1:N2]
tgt3 = [SVector{3,Float64}(3.0 + rand(), rand(), rand()) for _ in 1:N2]

K3 = KernelMatrix(src3, tgt3) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end

Xclt3 = ClusterTree(deepcopy(src3), GeometricSplitter(; nmax=20))
Yclt3 = ClusterTree(deepcopy(tgt3), GeometricSplitter(; nmax=20))

h2_3d = assemble_h2matrix(K3, Xclt3, Yclt3; order=3, global_index=true)

K3_dense = Matrix{Float64}(undef, N2, N2)
for j in 1:N2, i in 1:N2
    K3_dense[i, j] = K3[i, j]
end

x3 = randn(N2)
y_3d = h2_3d * x3
y_ref3 = K3_dense * x3
err_3d = norm(y_3d - y_ref3) / norm(y_ref3)
cr_3d = H2Matrices.compression_ratio(h2_3d)
println("3D Laplace (order=3): error = $(round(err_3d; sigdigits=3))")
println("  compression ratio = $(round(cr_3d; sigdigits=3))")

# --- Ex2 Plot: entry-wise error scatter ---
# Compare first 50 entries of matvec
nshow = min(50, N2)
p2 = plot(1:nshow, y_ref3[1:nshow], label="Dense (exact)", lw=2,
    title="Example 2: Matvec Comparison (3D Laplace)",
    xlabel="Entry index", ylabel="y = K * x", size=(650, 350))
scatter!(p2, 1:nshow, y_3d[1:nshow], label="H²-matrix", ms=4, mc=:darkorange, alpha=0.8)
savefig(p2, joinpath(assets, "ex2_matvec.png"))
println("Saved ex2_matvec.png")

# =========================================================================
# Example 3: Adaptive Assembly with Automatic Tree
# =========================================================================
println("\n" * "="^60)
println("EXAMPLE 3: Adaptive Assembly (auto tree)")
println("="^60)

Random.seed!(42)
N3 = 200
src_a = [SVector{2,Float64}(rand(), rand()) for _ in 1:N3]
tgt_a = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:N3]
K_a = KernelMatrix(src_a, tgt_a) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end

h2_auto = assemble_h2matrix_adaptive(K_a; rtol=1e-6, maxrank=50, nmax=32)
println("Size: ", size(h2_auto))
x_a = randn(N3)
y_a = h2_auto * x_a
println("Matvec computed with $(length(y_a)) entries")
cr_auto = H2Matrices.compression_ratio(h2_auto)
println("Compression ratio: $(round(cr_auto; sigdigits=3))")

# Dense reference for error
K_a_dense = Matrix{Float64}(undef, N3, N3)
for j in 1:N3, i in 1:N3
    K_a_dense[i, j] = K_a[i, j]
end
y_ref_a = K_a_dense * x_a
err_auto = norm(y_a - y_ref_a) / norm(y_ref_a)
println("Relative matvec error: $(round(err_auto; sigdigits=3))")

# --- Ex3 Plot: point geometry with src/tgt shown ---
p3 = scatter([s[1] for s in src_a], [s[2] for s in src_a],
    label="Source", ms=3, mc=:steelblue, alpha=0.7,
    title="Example 3: Point Geometry", xlabel="x", ylabel="y",
    size=(600, 350), aspect_ratio=:auto)
scatter!(p3, [t[1] for t in tgt_a], [t[2] for t in tgt_a],
    label="Target", ms=3, mc=:darkorange, alpha=0.7)
savefig(p3, joinpath(assets, "ex3_points.png"))
println("Saved ex3_points.png")

# =========================================================================
# Example 4: H-Matrix to H²-Matrix Conversion
# =========================================================================
println("\n" * "="^60)
println("EXAMPLE 4: H-Matrix → H²-Matrix Conversion")
println("="^60)

Random.seed!(42)
N4 = 300
src4 = [SVector{2,Float64}(rand(), rand()) for _ in 1:N4]
tgt4 = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:N4]
K4 = KernelMatrix(src4, tgt4) do x, y
    r = norm(x - y)
    r > 0 ? 1 / (4π * r) : 0.0
end

Xclt4 = ClusterTree(deepcopy(src4), GeometricSplitter(; nmax=30))
Yclt4 = ClusterTree(deepcopy(tgt4), GeometricSplitter(; nmax=30))

hmat4 = assemble_hmatrix(K4, Xclt4, Yclt4;
    comp=PartialACA(; rtol=1e-10),
    global_index=true, threads=false)

h2_conv = compress_hmatrix_to_h2(hmat4; rtol=1e-6, maxrank=50)
cr_before_recomp = H2Matrices.compression_ratio(h2_conv)
rank_before4 = H2Matrices.total_rank(h2_conv.row_basis)

recompress!(h2_conv; rtol=1e-4, maxrank=30)
cr_after_recomp = H2Matrices.compression_ratio(h2_conv)
rank_after4 = H2Matrices.total_rank(h2_conv.row_basis)

println("H²-matrix size: ", size(h2_conv))
println("Before recompress: compression ratio = $(round(cr_before_recomp; sigdigits=3)), total rank = $rank_before4")
println("After  recompress: compression ratio = $(round(cr_after_recomp; sigdigits=3)), total rank = $rank_after4")

# Dense reference for error
K4_dense = Matrix{Float64}(undef, N4, N4)
for j in 1:N4, i in 1:N4
    K4_dense[i, j] = K4[i, j]
end
x4 = randn(N4)
y_ref4 = K4_dense * x4
err4 = norm(h2_conv * x4 - y_ref4) / norm(y_ref4)
println("Matvec error after recompression: $(round(err4; sigdigits=3))")

# --- Ex4 Plot: rank reduction bar chart ---
p4 = bar(["Before\nRecompression", "After\nRecompression"],
    [rank_before4, rank_after4],
    ylabel="Total Basis Rank", title="Example 4: Rank Reduction via Recompression",
    legend=false, color=[:steelblue, :seagreen],
    bar_width=0.5, size=(500, 350))
savefig(p4, joinpath(assets, "ex4_recompression.png"))
println("Saved ex4_recompression.png")

# =========================================================================
# Example 5: Large-Scale 3D Problem (Sphere) — scaled down for docs
# =========================================================================
println("\n" * "="^60)
println("EXAMPLE 5: 3D Sphere (scaled to m=10000 for docs)")
println("="^60)

Random.seed!(42)
const Point3D = SVector{3,Float64}

# Use m=10_000 for docs (the real example says 100_000)
m = 10_000
X5 = Y5 = [Point3D(sin(θ)cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
            for (θ,ϕ) in zip(π*rand(m), 2π*rand(m))]

function Gfun(x, y)
    d = norm(x - y) + 1e-8
    1 / (4π * d)
end

K5 = KernelMatrix(Gfun, X5, Y5)

# H-matrix
H5 = assemble_hmatrix(K5; atol=1e-6)
cr_h5 = HMatrices.compression_ratio(H5)
println("H-matrix compression ratio: $(round(cr_h5; sigdigits=3))")

x5 = rand(m)
y_h5 = H5 * x5

# H²-matrix
h2_5 = assemble_h2matrix_adaptive(K5; rtol=1e-6, maxrank=80, nmax=32)
cr_h2_5 = H2Matrices.compression_ratio(h2_5)
println("H²-matrix compression ratio: $(round(cr_h2_5; sigdigits=3))")

y_h2_5 = h2_5 * x5

# Spot-check error at entry 42
exact_42 = sum(K5[42, j] * x5[j] for j in 1:m)
err_h_42 = abs(y_h5[42] - exact_42)
err_h2_42 = abs(y_h2_5[42] - exact_42)
println("H  matvec y[42] error: $(round(err_h_42; sigdigits=3))")
println("H² matvec y[42] error: $(round(err_h2_42; sigdigits=3))")

# --- Ex5 Plot: 3D scatter of sphere points + compression bar chart ---
p5a = scatter([p[1] for p in X5], [p[2] for p in X5], [p[3] for p in X5],
    ms=1, mc=:steelblue, alpha=0.3, label="",
    title="$m points on a sphere", xlabel="x", ylabel="y", zlabel="z",
    size=(550, 450), camera=(30, 30))
savefig(p5a, joinpath(assets, "ex5_sphere.png"))
println("Saved ex5_sphere.png")

p5b = bar(["H-matrix", "H²-matrix"],
    [cr_h5, cr_h2_5],
    ylabel="Compression Ratio", title="Example 5: H vs H² Compression (m=$m)",
    legend=false, color=[:steelblue, :seagreen],
    bar_width=0.5, size=(500, 350))
savefig(p5b, joinpath(assets, "ex5_compression.png"))
println("Saved ex5_compression.png")

println("\n✓ All examples completed.")
