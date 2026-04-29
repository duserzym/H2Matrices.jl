#!/usr/bin/env julia
"""
Generate figures for H2Matrices.jl documentation.

Run from the repo root:
    julia docs/generate_figures.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using H2Matrices
using HMatrices
using HMatrices: ClusterTree, KernelMatrix, GeometricSplitter,
    StrongAdmissibilityStd, PartialACA,
    index_range, loc2glob, assemble_hmatrix
using StaticArrays, LinearAlgebra, Random
using Plots
using Plots: cgrad, RGB

const ASSETS = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS)

const DENSE_COLOR = RGB(0.85, 0.32, 0.15)
const LOW_RANK_COLOR = RGB(0.04, 0.17, 0.42)
const HIGH_RANK_COLOR = RGB(1.00, 0.77, 0.16)

function rank_cgrad(maxrank)
    maxrank = max(maxrank, 1)
    maxrank == 1 && return cgrad([DENSE_COLOR, LOW_RANK_COLOR], [-1, 1])
    cgrad([DENSE_COLOR, LOW_RANK_COLOR, HIGH_RANK_COLOR], [-1, 1, maxrank])
end

# ──────────────────────────────────────────────────────────────────
# Setup: 2D Laplace problem (coincident geometry for block structure)
# ──────────────────────────────────────────────────────────────────
Random.seed!(42)
N = 500
pts = [SVector{2,Float64}(rand(), rand()) for _ in 1:N]

K = KernelMatrix(pts, pts) do x, y
    r = norm(x - y) + 1e-10
    1 / (4π * r)
end

Xclt = ClusterTree(deepcopy(pts), GeometricSplitter(; nmax=32))
Yclt = ClusterTree(deepcopy(pts), GeometricSplitter(; nmax=32))

# Dense reference in local ordering
K_dense = Matrix{Float64}(undef, N, N)
rp, cp = loc2glob(Xclt), loc2glob(Yclt)
for j in 1:N, i in 1:N
    K_dense[i, j] = K[rp[i], cp[j]]
end

# ──────────────────────────────────────────────────────────────────
# Figure 1: Dense kernel matrix heatmap
# ──────────────────────────────────────────────────────────────────
println("Generating dense kernel heatmap...")
p1 = heatmap(log10.(abs.(K_dense) .+ 1e-16);
    yflip=true, color=:viridis, clims=(-3.0, 0.5),
    xlabel="Column index", ylabel="Row index",
    title="Dense Kernel Matrix (log₁₀|K|)",
    size=(500, 450), dpi=200, aspect_ratio=:equal,
    tickfontsize=8, guidefontsize=10, titlefontsize=11)
savefig(p1, joinpath(ASSETS, "dense_kernel.png"))
println("  → dense_kernel.png")

# ──────────────────────────────────────────────────────────────────
# Figure 2: H-matrix block structure (spy plot)
# ──────────────────────────────────────────────────────────────────
println("Generating H-matrix block structure...")

hmat = assemble_hmatrix(K, Xclt, Yclt;
    comp=PartialACA(; rtol=1e-8),
    global_index=true, threads=false)

"""Paint H-matrix blocks: admissible → rank value (>0), dense → -1."""
function paint_hmatrix!(img, hmat)
    if HMatrices.isleaf(hmat)
        rt = HMatrices.rowtree(hmat)
        ct = HMatrices.coltree(hmat)
        ri = index_range(rt)
        ci = index_range(ct)
        if HMatrices.isadmissible(hmat)
            d = HMatrices.data(hmat)
            rk = d === nothing ? 0 : size(d.A, 2)
            img[ri, ci] .= max(rk, 1)
        else
            img[ri, ci] .= -1
        end
    else
        for child in HMatrices.children(hmat)
            paint_hmatrix!(img, child)
        end
    end
end

"""Collect leaf-block boundaries."""
function block_boundaries_hmat!(shapes, hmat; admissible_only=false)
    if HMatrices.isleaf(hmat)
        if admissible_only && !HMatrices.isadmissible(hmat)
            return
        end
        rt = HMatrices.rowtree(hmat)
        ct = HMatrices.coltree(hmat)
        ri = index_range(rt)
        ci = index_range(ct)
        x1, x2 = ci.start - 0.5, ci.stop + 0.5
        y1, y2 = ri.start - 0.5, ri.stop + 0.5
        push!(shapes, (x1, x2, y1, y2, HMatrices.isadmissible(hmat)))
    else
        for child in HMatrices.children(hmat)
            block_boundaries_hmat!(shapes, child; admissible_only)
        end
    end
end

img_h = zeros(Float64, N, N)
paint_hmatrix!(img_h, hmat)

shapes_h = Tuple{Float64,Float64,Float64,Float64,Bool}[]
block_boundaries_hmat!(shapes_h, hmat)

p2 = plot(; size=(500, 480), dpi=200, aspect_ratio=:equal,
    xlabel="Column index", ylabel="Row index",
    title="H-Matrix Block Structure",
    tickfontsize=8, guidefontsize=10, titlefontsize=11,
    xlims=(0.5, N + 0.5), ylims=(0.5, N + 0.5), yflip=true,
    framestyle=:box)
# Plot the RGB image as a heatmap by using the block map values
heatmap!(p2, 1:N, 1:N, img_h'; yflip=true,
    color=rank_cgrad(maximum(img_h)),
    clims=(-1, max(maximum(img_h), 1)),
    colorbar=false)
for (x1, x2, y1, y2, _) in shapes_h
    plot!(p2, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1];
        color=:black, linewidth=0.4, label=false)
end
scatter!(p2, [-10], [-10]; color=LOW_RANK_COLOR, markershape=:rect,
    markersize=8, label="Low rank")
scatter!(p2, [-10], [-10]; color=HIGH_RANK_COLOR, markershape=:rect,
    markersize=8, label="Higher rank")
scatter!(p2, [-10], [-10]; color=DENSE_COLOR, markershape=:rect,
    markersize=8, label="Dense near-field")
savefig(p2, joinpath(ASSETS, "hmatrix_structure.png"))
println("  → hmatrix_structure.png")

# ──────────────────────────────────────────────────────────────────
# Figure 3: H²-matrix block structure
# ──────────────────────────────────────────────────────────────────
println("Generating H²-matrix block structure...")

h2 = compress_hmatrix_to_h2(hmat; rtol=1e-6, maxrank=50)

"""Paint H²-matrix blocks: uniform → rank value (>0), dense → -1."""
function paint_h2matrix!(img, h2)
    if H2Matrices.isleaf(h2)
        rb = h2.row_basis
        cb_col = h2.col_basis
        ri = index_range(rb.cluster)
        ci = index_range(cb_col.cluster)
        if H2Matrices.isadmissible(h2) && h2.uniform !== nothing
            rk = max(size(h2.uniform.S, 1), 1)
            img[ri, ci] .= rk
        elseif h2.dense !== nothing
            img[ri, ci] .= -1
        end
    else
        for child in h2.children
            paint_h2matrix!(img, child)
        end
    end
end

"""Collect H²-matrix block boundaries."""
function block_boundaries_h2!(shapes, h2)
    if H2Matrices.isleaf(h2)
        rb = h2.row_basis
        cb_col = h2.col_basis
        ri = index_range(rb.cluster)
        ci = index_range(cb_col.cluster)
        x1, x2 = ci.start - 0.5, ci.stop + 0.5
        y1, y2 = ri.start - 0.5, ri.stop + 0.5
        push!(shapes, (x1, x2, y1, y2, H2Matrices.isadmissible(h2)))
    else
        for child in h2.children
            block_boundaries_h2!(shapes, child)
        end
    end
end

img_h2 = zeros(Float64, N, N)
paint_h2matrix!(img_h2, h2)

shapes_h2 = Tuple{Float64,Float64,Float64,Float64,Bool}[]
block_boundaries_h2!(shapes_h2, h2)

p3 = plot(; size=(500, 480), dpi=200, aspect_ratio=:equal,
    xlabel="Column index", ylabel="Row index",
    title="H²-Matrix Block Structure (nested bases)",
    tickfontsize=8, guidefontsize=10, titlefontsize=11,
    xlims=(0.5, N + 0.5), ylims=(0.5, N + 0.5), yflip=true,
    framestyle=:box)
heatmap!(p3, 1:N, 1:N, img_h2'; yflip=true,
    color=rank_cgrad(maximum(img_h2)),
    clims=(-1, max(maximum(img_h2), 1)),
    colorbar=false)
for (x1, x2, y1, y2, _) in shapes_h2
    plot!(p3, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1];
        color=:black, linewidth=0.4, label=false)
end
scatter!(p3, [-10], [-10]; color=LOW_RANK_COLOR, markershape=:rect,
    markersize=8, label="Low-rank uniform")
scatter!(p3, [-10], [-10]; color=HIGH_RANK_COLOR, markershape=:rect,
    markersize=8, label="Higher-rank uniform")
scatter!(p3, [-10], [-10]; color=DENSE_COLOR, markershape=:rect,
    markersize=8, label="Dense near-field")
savefig(p3, joinpath(ASSETS, "h2matrix_structure.png"))
println("  → h2matrix_structure.png")

# ──────────────────────────────────────────────────────────────────
# Figure 4: Side-by-side comparison (H vs H²)
# ──────────────────────────────────────────────────────────────────
println("Generating H vs H² comparison...")

p4 = plot(p2, p3; layout=(1, 2), size=(1050, 480), dpi=200,
    plot_title="H-Matrix vs H²-Matrix Compression")
savefig(p4, joinpath(ASSETS, "h_vs_h2_comparison.png"))
println("  → h_vs_h2_comparison.png")
savefig(p4, joinpath(ASSETS, "h_and_h2_matrix_block_structures.png"))
println("  → h_and_h2_matrix_block_structures.png")

# ──────────────────────────────────────────────────────────────────
# Figure 5: Approximation error heatmaps
# ──────────────────────────────────────────────────────────────────
println("Generating error heatmaps...")

# H²-matrix (Chebyshev, coincident setup)
h2_cheb = assemble_h2matrix(K, Xclt, Yclt; order=4, global_index=true)
M_cheb = Matrix(h2_cheb)
err_cheb = abs.(M_cheb .- K_dense)

# H²-matrix (adaptive, already built as h2)
M_ada = Matrix(h2)
err_ada = abs.(M_ada .- K_dense)

p5a = heatmap(log10.(err_cheb .+ 1e-16); yflip=true, color=:hot,
    clims=(-10, 0), xlabel="Column", ylabel="Row",
    title="Chebyshev H² error (log₁₀)",
    size=(500, 450), dpi=200, aspect_ratio=:equal,
    tickfontsize=8, guidefontsize=10, titlefontsize=10)

p5b = heatmap(log10.(err_ada .+ 1e-16); yflip=true, color=:hot,
    clims=(-10, 0), xlabel="Column", ylabel="Row",
    title="Adaptive H² error (log₁₀)",
    size=(500, 450), dpi=200, aspect_ratio=:equal,
    tickfontsize=8, guidefontsize=10, titlefontsize=10)

p5 = plot(p5a, p5b; layout=(1, 2), size=(1050, 480), dpi=200,
    plot_title="Element-wise Approximation Error")
savefig(p5, joinpath(ASSETS, "error_heatmaps.png"))
println("  → error_heatmaps.png")

# ──────────────────────────────────────────────────────────────────
# Figure 6: Point geometry
# ──────────────────────────────────────────────────────────────────
println("Generating point geometry plot...")

p6 = scatter(getindex.(pts, 1), getindex.(pts, 2);
    color="#1f5fa6", markersize=2.5, markerstrokewidth=0,
    label="Points (source = target)", aspect_ratio=:equal,
    xlabel="x", ylabel="y", title="Point Distribution (N=$N, 2D Laplace)",
    size=(500, 450), dpi=200,
    tickfontsize=8, guidefontsize=10, titlefontsize=11)
savefig(p6, joinpath(ASSETS, "point_geometry.png"))
println("  → point_geometry.png")

# ──────────────────────────────────────────────────────────────────
# Figure 7: Recompression rank reduction
# ──────────────────────────────────────────────────────────────────
println("Generating recompression comparison...")

h2_before = assemble_h2matrix(K, Xclt, Yclt; order=5, global_index=true)
img_before = zeros(Float64, N, N)
paint_h2matrix!(img_before, h2_before)
rank_before = H2Matrices.total_rank(h2_before.row_basis)

recompress!(h2_before; rtol=1e-4, maxrank=50)
img_after = zeros(Float64, N, N)
paint_h2matrix!(img_after, h2_before)
rank_after = H2Matrices.total_rank(h2_before.row_basis)

max_rk_recomp = max(maximum(img_before), maximum(img_after), 1)

p7a = plot(; size=(500, 480), aspect_ratio=:equal, framestyle=:box,
    xlabel="Column", ylabel="Row",
    title="Before (total rank = $rank_before)",
    xlims=(0.5, N+0.5), ylims=(0.5, N+0.5), yflip=true,
    tickfontsize=8, guidefontsize=10, titlefontsize=10)
heatmap!(p7a, 1:N, 1:N, img_before'; yflip=true,
    color=rank_cgrad(max_rk_recomp),
    clims=(-1, max_rk_recomp), colorbar=false)

p7b = plot(; size=(500, 480), aspect_ratio=:equal, framestyle=:box,
    xlabel="Column", ylabel="Row",
    title="After (total rank = $rank_after)",
    xlims=(0.5, N+0.5), ylims=(0.5, N+0.5), yflip=true,
    tickfontsize=8, guidefontsize=10, titlefontsize=10)
heatmap!(p7b, 1:N, 1:N, img_after'; yflip=true,
    color=rank_cgrad(max_rk_recomp),
    clims=(-1, max_rk_recomp), colorbar=false)

p7 = plot(p7a, p7b; layout=(1, 2), size=(1050, 480), dpi=200,
    plot_title="H² Recompression: Rank Reduction")
savefig(p7, joinpath(ASSETS, "recompression.png"))
println("  → recompression.png")

println("\nAll figures generated in $ASSETS")
