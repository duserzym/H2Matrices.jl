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
    include("test_h2lib_parity.jl")
    include("test_solvers.jl")
end
