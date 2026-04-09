"""
    mutable struct ClusterBasis{N,T}

Hierarchical cluster basis for H²-matrices. Each node stores either a leaf
basis matrix `V` (for leaf clusters) or a transfer matrix `E` that maps
child basis coefficients to the parent level.

For a leaf cluster `t`, the basis is represented by `V_t ∈ ℝ^{|t| × k}`.
For a non-leaf cluster `t` with children `t₁, ..., tₘ`, the basis satisfies:

    V_t = [V_{t₁}; V_{t₂}; ...] * blkdiag(E_{t₁}, E_{t₂}, ...)

where `E_{tᵢ} ∈ ℝ^{k_child × k_parent}` are the transfer matrices.

# Fields
- `cluster::ClusterTree{N,T}` : the underlying cluster tree node
- `k::Int` : rank (number of basis vectors) at this node
- `V::Matrix{Float64}` : leaf basis matrix (only for leaf nodes, size |t| × k)
- `E::Matrix{Float64}` : transfer matrix to parent (size k_self × k_parent)
- `children::Vector{ClusterBasis{N,T}}` : child cluster bases
- `parent::ClusterBasis{N,T}` : parent cluster basis
"""
mutable struct ClusterBasis{N,T}
    cluster::ClusterTree{N,T}
    k::Int                              # rank at this level
    V::Matrix{Float64}                  # leaf basis (|t| × k), empty for non-leaves
    E::Matrix{Float64}                  # transfer matrix (k_self × k_parent), empty for root
    children::Vector{ClusterBasis{N,T}}
    parent::ClusterBasis{N,T}

    function ClusterBasis(cluster::ClusterTree{N,T}) where {N,T}
        cb = new{N,T}()
        cb.cluster = cluster
        cb.k = 0
        cb.V = Matrix{Float64}(undef, 0, 0)
        cb.E = Matrix{Float64}(undef, 0, 0)
        cb.children = ClusterBasis{N,T}[]
        cb.parent = cb  # self-referential for root
        return cb
    end
end

# Accessors
cluster(cb::ClusterBasis) = cb.cluster
rank(cb::ClusterBasis) = cb.k
leaf_basis(cb::ClusterBasis) = cb.V
transfer_matrix(cb::ClusterBasis) = cb.E
children(cb::ClusterBasis) = cb.children
parentnode(cb::ClusterBasis) = cb.parent
isleaf(cb::ClusterBasis) = isempty(cb.children)
isroot(cb::ClusterBasis) = cb.parent === cb
Base.length(cb::ClusterBasis) = length(index_range(cb.cluster))

"""
    build_cluster_basis(clt::ClusterTree)

Build a `ClusterBasis` tree that mirrors the structure of the given
`ClusterTree`, with zero rank and empty matrices. The basis matrices are
filled later during assembly.
"""
function build_cluster_basis(clt::ClusterTree{N,T}) where {N,T}
    root = ClusterBasis(clt)
    _build_cluster_basis_recursive!(root)
    return root
end

function _build_cluster_basis_recursive!(cb::ClusterBasis{N,T}) where {N,T}
    clt = cb.cluster
    if !HMatrices.isleaf(clt)
        for child_clt in HMatrices.children(clt)
            child_cb = ClusterBasis(child_clt)
            child_cb.parent = cb
            push!(cb.children, child_cb)
            _build_cluster_basis_recursive!(child_cb)
        end
    end
    return cb
end

"""
    set_leaf_basis!(cb::ClusterBasis, V::Matrix)

Set the leaf basis matrix `V` for a leaf cluster basis node.
"""
function set_leaf_basis!(cb::ClusterBasis, V::Matrix{Float64})
    @assert isleaf(cb) "Can only set leaf basis on leaf nodes"
    cb.V = V
    cb.k = size(V, 2)
    return cb
end

"""
    set_transfer_matrix!(cb::ClusterBasis, E::Matrix)

Set the transfer matrix `E` for a non-root cluster basis node.
"""
function set_transfer_matrix!(cb::ClusterBasis, E::Matrix{Float64})
    cb.E = E
    cb.k = size(E, 1)
    return cb
end

"""
    leaves(cb::ClusterBasis)

Return all leaf nodes of the cluster basis tree.
"""
function leaves(cb::ClusterBasis{N,T}) where {N,T}
    result = ClusterBasis{N,T}[]
    _collect_leaves!(result, cb)
    return result
end

function _collect_leaves!(result, cb::ClusterBasis)
    if isleaf(cb)
        push!(result, cb)
    else
        for child in cb.children
            _collect_leaves!(result, child)
        end
    end
    return result
end

"""
    nodes(cb::ClusterBasis)

Return all nodes of the cluster basis tree.
"""
function nodes(cb::ClusterBasis{N,T}) where {N,T}
    result = ClusterBasis{N,T}[]
    _collect_nodes!(result, cb)
    return result
end

function _collect_nodes!(result, cb::ClusterBasis)
    push!(result, cb)
    for child in cb.children
        _collect_nodes!(result, child)
    end
    return result
end

"""
    total_rank(cb::ClusterBasis)

Sum of ranks across entire subtree (ktree in H2Lib).
"""
function total_rank(cb::ClusterBasis)
    s = cb.k
    for child in cb.children
        s += total_rank(child)
    end
    return s
end
