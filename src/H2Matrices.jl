module H2Matrices

using LinearAlgebra
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
include("assembly.jl")
include("compression.jl")
include("plotting.jl")

export ClusterBasis,
    H2Matrix,
    UniformBlock,
    assemble_h2matrix,
    assemble_h2matrix_adaptive,
    compress_hmatrix_to_h2,
    compression_ratio,
    depth,
    recompress!,
    forward_transform!,
    backward_transform!,
    h2matvec!

end # module
