"""
    mutable struct H2Matrix{N,T} <: AbstractMatrix{Float64}

An H²-matrix built from a row and column cluster tree with nested
cluster bases. The key difference from a standard H-matrix is that
admissible (far-field) blocks share cluster bases across the tree,
enabling O(N) storage and matrix-vector products.

The matrix has the same block structure as an H-matrix, but:
- Admissible blocks store only a small coupling matrix `S` (via `UniformBlock`)
- Inadmissible (near-field) blocks store a dense matrix
- Non-leaf blocks recurse into children

# Fields
- `row_basis::ClusterBasis{N,T}` : row cluster basis for this block
- `col_basis::ClusterBasis{N,T}` : column cluster basis for this block
- `uniform::Union{UniformBlock{N,T}, Nothing}` : uniform block for admissible leaves
- `dense::Union{Matrix{Float64}, Nothing}` : dense block for inadmissible leaves
- `children::Matrix{H2Matrix{N,T}}` : child blocks (rsons × csons)
- `admissible::Bool` : whether this block is admissible
"""
mutable struct H2Matrix{N,T} <: AbstractMatrix{Float64}
    row_basis::ClusterBasis{N,T}
    col_basis::ClusterBasis{N,T}
    uniform::Union{UniformBlock{N,T}, Nothing}
    dense::Union{Matrix{Float64}, Nothing}
    children::Matrix{H2Matrix{N,T}}
    admissible::Bool

    function H2Matrix(rb::ClusterBasis{N,T}, cb::ClusterBasis{N,T}) where {N,T}
        h2 = new{N,T}()
        h2.row_basis = rb
        h2.col_basis = cb
        h2.uniform = nothing
        h2.dense = nothing
        h2.children = Matrix{H2Matrix{N,T}}(undef, 0, 0)
        h2.admissible = false
        return h2
    end
end

# Accessors
row_basis(h::H2Matrix) = h.row_basis
col_basis(h::H2Matrix) = h.col_basis
isleaf(h::H2Matrix) = isempty(h.children)
isadmissible(h::H2Matrix) = h.admissible
hasdata(h::H2Matrix) = h.uniform !== nothing || h.dense !== nothing
children(h::H2Matrix) = h.children

function Base.size(h::H2Matrix)
    rc = h.row_basis.cluster
    cc = h.col_basis.cluster
    return (length(index_range(rc)), length(index_range(cc)))
end

Base.eltype(::H2Matrix) = Float64

"""
    leaves(h::H2Matrix)

Return all leaf blocks of the H2Matrix.
"""
function leaves(h::H2Matrix{N,T}) where {N,T}
    result = H2Matrix{N,T}[]
    _collect_h2_leaves!(result, h)
    return result
end

function _collect_h2_leaves!(result, h::H2Matrix)
    if isleaf(h)
        push!(result, h)
    else
        for child in h.children
            _collect_h2_leaves!(result, child)
        end
    end
    return result
end

"""
    nodes(h::H2Matrix)

Return all nodes of the H2Matrix tree.
"""
function nodes(h::H2Matrix{N,T}) where {N,T}
    result = H2Matrix{N,T}[]
    _collect_h2_nodes!(result, h)
    return result
end

function _collect_h2_nodes!(result, h::H2Matrix)
    push!(result, h)
    for child in h.children
        _collect_h2_nodes!(result, child)
    end
    return result
end

"""
    build_h2_block_structure(rb, cb, adm)

Build the block structure for an H²-matrix given row/column cluster bases
and an admissibility condition.
"""
function build_h2_block_structure(
    rb::ClusterBasis{N,T},
    cb::ClusterBasis{N,T},
    adm
) where {N,T}
    root = H2Matrix(rb, cb)
    _build_h2_blocks!(root, adm)
    return root
end

function _build_h2_blocks!(h2::H2Matrix{N,T}, adm) where {N,T}
    rc = h2.row_basis.cluster
    cc = h2.col_basis.cluster
    if HMatrices.isleaf(rc) || HMatrices.isleaf(cc)
        # Leaf cluster: check admissibility
        h2.admissible = adm(rc, cc)
    elseif adm(rc, cc)
        # Admissible non-leaf: store as uniform block
        h2.admissible = true
    else
        # Not admissible: subdivide
        h2.admissible = false
        row_children = h2.row_basis.children
        col_children = h2.col_basis.children
        nr = length(row_children)
        nc = length(col_children)
        h2.children = Matrix{H2Matrix{N,T}}(undef, nr, nc)
        for i in 1:nr
            for j in 1:nc
                child = H2Matrix(row_children[i], col_children[j])
                h2.children[i, j] = child
                _build_h2_blocks!(child, adm)
            end
        end
    end
    return h2
end

"""
    compression_ratio(h::H2Matrix)

The ratio of uncompressed size to compressed storage.
"""
function compression_ratio(h::H2Matrix)
    m, n = size(h)
    uncompressed = m * n * sizeof(Float64)
    compressed = _storage_bytes(h)
    return uncompressed / compressed
end

function _storage_bytes(h::H2Matrix)
    if isleaf(h)
        if h.uniform !== nothing
            return sizeof(h.uniform.S)
        elseif h.dense !== nothing
            return sizeof(h.dense)
        else
            return 0
        end
    else
        s = 0
        for child in h.children
            s += _storage_bytes(child)
        end
        return s
    end
end

"""
    Matrix(h::H2Matrix; global_index=false)

Convert an H²-matrix to a dense matrix.
"""
function Base.Matrix(h::H2Matrix; global_index=false)
    m, n = size(h)
    M = zeros(Float64, m, n)
    rc = h.row_basis.cluster
    cc = h.col_basis.cluster
    row_offset = index_range(rc).start - 1
    col_offset = index_range(cc).start - 1
    _fill_dense!(M, h, row_offset, col_offset)
    if global_index
        rp = loc2glob(rc)
        cp = loc2glob(cc)
        # Permute from local to global ordering
        M_global = zeros(Float64, m, n)
        for j in 1:n
            for i in 1:m
                M_global[rp[i], cp[j]] = M[i, j]
            end
        end
        return M_global
    end
    return M
end

function _fill_dense!(M, h::H2Matrix, row_offset, col_offset)
    if isleaf(h)
        rc = h.row_basis.cluster
        cc = h.col_basis.cluster
        irange = index_range(rc)
        jrange = index_range(cc)
        local_i = irange .- row_offset
        local_j = jrange .- col_offset
        if h.uniform !== nothing
            M[local_i, local_j] .= Matrix(h.uniform)
        elseif h.dense !== nothing
            M[local_i, local_j] .= h.dense
        end
    else
        for child in h.children
            _fill_dense!(M, child, row_offset, col_offset)
        end
    end
    return M
end

function Base.show(io::IO, h::H2Matrix{N,T}) where {N,T}
    m, n = size(h)
    print(io, "H2Matrix{$N,$T}: $m × $n")
    lvs = leaves(h)
    n_adm = count(isadmissible, lvs)
    n_dense = length(lvs) - n_adm
    print(io, "\n  leaves: $(length(lvs)) ($n_adm admissible + $n_dense dense)")
    print(io, "\n  compression ratio: $(round(compression_ratio(h); digits=2))")
end
Base.show(io::IO, ::MIME"text/plain", h::H2Matrix) = show(io, h)
