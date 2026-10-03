using H2Matrices
using HMatrices
using HMatrices: ClusterTree, KernelMatrix, GeometricSplitter, PartialACA,
    assemble_hmatrix, index_range
using StaticArrays
using LinearAlgebra
using Random
using Test

include("support.jl")

@testset "H2Matrices.jl" begin
    include("test_basis.jl")
    include("test_assembly.jl")
    include("test_compression.jl")
    include("test_accuracy_regressions.jl")
    include("test_conversion.jl")
    include("test_matvec_plan.jl")
    include("test_compact_plan.jl")
    include("test_packet_plan.jl")
    include("test_global_control.jl")
    include("test_h2lib_parity.jl")
    include("test_solvers.jl")
end
