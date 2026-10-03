module H2Matrices

using LinearAlgebra
using Random
using StaticArrays
using HMatrices
using HMatrices: ClusterTree, HyperRectangle, AbstractKernelMatrix, KernelMatrix,
    HMatrix, PartialACA,
    index_range, container,
    root_elements, elements, loc2glob, glob2loc,
    diameter, distance, center, low_corner, high_corner,
    StrongAdmissibilityStd, GeometricSplitter,
    getblock!
using RecipesBase

include("clusterbasis.jl")
include("uniformblock.jl")
include("h2matrix.jl")
include("basis_construction.jl")
include("matvec.jl")
include("matvec_plan.jl")
include("diagnostics.jl")
storage_bytes(p::H2LowRankMatvecPlan) = _lowrank_plan_storage_bytes(p)
include("assembly.jl")
include("compression.jl")
include("condensed_conversion.jl")
include("copying.jl")
include("compact_plan.jl")
include("packet_plan.jl")
include("packet_multi.jl")
include("solvers.jl")
include("plotting.jl")

export H2PacketMatvecPlan,
    H2CompactMatvecPlan,
    H2LowRankMatvecPlan,
    H2MatvecPlan,
    ClusterBasis,
    H2Matrix,
    UniformBlock,
    assemble_h2matrix,
    assemble_h2matrix_adaptive,
    compress_hmatrix_to_h2,
    estimate_operator_scale,
    compression_ratio,
    compression_summary,
    storage_bytes,
    multi_workspace_bytes,
    release_multi_workspace!,
    dense_storage_bytes,
    block_stats,
    rank_stats,
    dense_reference,
    relative_matvec_error,
    sampled_frobenius_error,
    compress_matrix_to_h2,
    depth,
    H2SolveResult,
    solve_cg,
    solve_gmres,
    recompress!,
    forward_transform!,
    backward_transform!,
    h2matvec!

end # module
