"""
    struct UniformBlock{N,T}

Represents an admissible (far-field) block in an H²-matrix. The block
for cluster pair (τ, σ) is represented as:

    A|_{τ×σ} ≈ V_τ * S * W_σ'

where `V_τ` and `W_σ` are the row/column cluster bases, and `S` is
the small coupling matrix (size k_row × k_col).

# Fields
- `row_basis::ClusterBasis{N,T}` : row cluster basis
- `col_basis::ClusterBasis{N,T}` : column cluster basis
- `S::Matrix{Float64}` : coupling matrix (k_row × k_col)
"""
struct UniformBlock{N,T}
    row_basis::ClusterBasis{N,T}
    col_basis::ClusterBasis{N,T}
    S::Matrix{Float64}
end

row_basis(u::UniformBlock) = u.row_basis
col_basis(u::UniformBlock) = u.col_basis
coupling_matrix(u::UniformBlock) = u.S

Base.size(u::UniformBlock) = (length(u.row_basis), length(u.col_basis))
Base.eltype(::UniformBlock) = Float64

"""
    Matrix(u::UniformBlock)

Convert a uniform block to a dense matrix by expanding:
    A = V_τ * S * W_σ'
"""
function Base.Matrix(u::UniformBlock)
    V = _full_basis(u.row_basis)
    W = _full_basis(u.col_basis)
    return V * u.S * W'
end

"""
    _full_basis(cb::ClusterBasis)

Compute the full (expanded) basis matrix for a cluster basis node.
For a leaf, this is just `V`. For a non-leaf, this recursively builds
the full basis via transfer matrices.
"""
function _full_basis(cb::ClusterBasis)
    if isleaf(cb)
        return cb.V
    else
        n = length(cb)
        k = cb.k
        V = zeros(n, k)
        offset = 0
        irange = index_range(cb.cluster)
        for child in cb.children
            child_range = index_range(child.cluster)
            # Map child range to local indices within parent
            local_start = child_range.start - irange.start + 1
            local_end = child_range.stop - irange.start + 1
            V_child = _full_basis(child)  # |child| × k_child
            # V_child * E_child gives |child| × k_parent
            V[local_start:local_end, :] = V_child * child.E
        end
        return V
    end
end
