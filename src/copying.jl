"""
    copy_cluster_basis(cb)

Deep-copy a cluster basis tree while preserving references to the underlying
`ClusterTree` nodes. This is useful for projection, recompression experiments,
and tests that need to keep an original H² matrix unchanged.
"""
function copy_cluster_basis(cb::ClusterBasis{N,T}) where {N,T}
    map = Dict{UInt,ClusterBasis{N,T}}()
    root = _copy_cluster_basis_recursive(cb, map)
    root.parent = root
    return root, map
end

function _copy_cluster_basis_recursive(cb::ClusterBasis{N,T},
                                       map::Dict{UInt,ClusterBasis{N,T}}) where {N,T}
    out = ClusterBasis(cb.cluster)
    out.k = cb.k
    out.V = copy(cb.V)
    out.E = copy(cb.E)
    map[objectid(cb)] = out
    empty!(out.children)
    for child in cb.children
        child_out = _copy_cluster_basis_recursive(child, map)
        child_out.parent = out
        push!(out.children, child_out)
    end
    return out
end

Base.copy(cb::ClusterBasis) = first(copy_cluster_basis(cb))

"""
    copy(h::H2Matrix)

Deep-copy an H² matrix, including all cluster basis data, dense near-field
blocks, coupling matrices, and block tree metadata.
"""
function Base.copy(h::H2Matrix{N,T}) where {N,T}
    rb, row_map = copy_cluster_basis(h.row_basis)
    cb, col_map = copy_cluster_basis(h.col_basis)
    return _copy_h2_recursive(h, row_map, col_map; root_bases=(rb, cb))
end

function _copy_h2_recursive(h::H2Matrix{N,T},
                            row_map::Dict{UInt,ClusterBasis{N,T}},
                            col_map::Dict{UInt,ClusterBasis{N,T}};
                            root_bases=nothing) where {N,T}
    rb = row_map[objectid(h.row_basis)]
    cb = col_map[objectid(h.col_basis)]
    out = H2Matrix(rb, cb; global_index=h.global_index)
    out.admissible = h.admissible
    out.uniform = h.uniform === nothing ? nothing : UniformBlock(rb, cb, copy(h.uniform.S))
    out.dense = h.dense === nothing ? nothing : copy(h.dense)
    if !isleaf(h)
        nr, nc = size(h.children)
        out.children = Matrix{H2Matrix{N,T}}(undef, nr, nc)
        for j in 1:nc, i in 1:nr
            out.children[i, j] = _copy_h2_recursive(h.children[i, j], row_map, col_map)
        end
    end
    return out
end

"""
    compress_matrix_to_h2(A, rowtree, coltree; rtol=1e-8, maxrank=50, adm=StrongAdmissibilityStd(3))

Approximate a dense matrix by an H² matrix on the supplied cluster trees. This
is primarily intended for validation and H2Lib-style dense-to-H² parity tests;
large production problems should prefer kernel assembly or H-matrix-to-H²
conversion.
"""
function compress_matrix_to_h2(
    A::AbstractMatrix,
    rowtree::ClusterTree{N,T},
    coltree::ClusterTree{N,T};
    rtol::Float64=1e-8,
    maxrank::Int=50,
    adm=StrongAdmissibilityStd(3),
    global_index::Bool=false,
    _print::Bool=true,
) where {N,T}
    size(A) == (length(index_range(rowtree)), length(index_range(coltree))) ||
        throw(DimensionMismatch("dense matrix size does not match cluster trees"))

    rb = build_cluster_basis(rowtree)
    cb = build_cluster_basis(coltree)
    h2_shape = build_h2_block_structure(rb, cb, adm)

    row_data = Dict{UInt,Vector{NamedTuple{(:A,:B),Tuple{Matrix{Float64},Matrix{Float64}}}}}()
    col_data = Dict{UInt,Vector{NamedTuple{(:A,:B),Tuple{Matrix{Float64},Matrix{Float64}}}}}()
    dense_blocks = Dict{Tuple{UInt,UInt},Matrix{Float64}}()
    rk_blocks = Dict{Tuple{UInt,UInt},NamedTuple{(:A,:B),Tuple{Matrix{Float64},Matrix{Float64}}}}()

    _collect_dense_matrix_blocks!(row_data, col_data, dense_blocks, rk_blocks,
                                  h2_shape, A; rtol, maxrank)

    _build_adaptive_basis_recursive!(rb, row_data, Tuple{Matrix{Float64},UnitRange{Int}}[];
                                     rtol, maxrank, is_row=true)
    _build_adaptive_basis_recursive!(cb, col_data, Tuple{Matrix{Float64},UnitRange{Int}}[];
                                     rtol, maxrank, is_row=false)

    h2 = build_h2_block_structure(rb, cb, adm)
    h2.global_index = global_index
    _fill_h2_from_dense_blocks!(h2, dense_blocks, rk_blocks)
    _print && _print_compression_summary(h2)
    return h2
end

function _collect_dense_matrix_blocks!(row_data, col_data, dense_blocks, rk_blocks,
                                       h2::H2Matrix, A; rtol, maxrank)
    if isleaf(h2)
        rb = h2.row_basis
        cb = h2.col_basis
        ir = index_range(rb.cluster)
        jr = index_range(cb.cluster)
        key = (objectid(rb.cluster), objectid(cb.cluster))
        block = Matrix{Float64}(A[ir, jr])
        if isadmissible(h2)
            F = svd(block)
            k = _truncation_rank(F.S, rtol, maxrank)
            if k == 0
                entry = (A=zeros(Float64, length(ir), 0), B=zeros(Float64, length(jr), 0))
            else
                U = F.U[:, 1:k]
                V = F.V[:, 1:k]
                entry = (A=U * Diagonal(F.S[1:k]), B=V)
            end
            rk_blocks[key] = entry
            push!(get!(row_data, objectid(rb.cluster), valtype(row_data)()), entry)
            push!(get!(col_data, objectid(cb.cluster), valtype(col_data)()), entry)
        else
            dense_blocks[key] = block
        end
    else
        for child in h2.children
            _collect_dense_matrix_blocks!(row_data, col_data, dense_blocks, rk_blocks,
                                          child, A; rtol, maxrank)
        end
    end
    return nothing
end

function _fill_h2_from_dense_blocks!(h2::H2Matrix, dense_blocks, rk_blocks)
    if isleaf(h2)
        rb = h2.row_basis
        cb = h2.col_basis
        key = (objectid(rb.cluster), objectid(cb.cluster))
        if isadmissible(h2)
            entry = rk_blocks[key]
            VA = _compress_basis_matrix(rb, entry.A)
            WB = _compress_basis_matrix(cb, entry.B)
            h2.uniform = UniformBlock(rb, cb, VA * WB')
        else
            h2.dense = dense_blocks[key]
        end
    else
        for child in h2.children
            _fill_h2_from_dense_blocks!(child, dense_blocks, rk_blocks)
        end
    end
    return h2
end
