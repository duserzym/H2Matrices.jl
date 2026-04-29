"""
    storage_bytes(h; include_bases=true)

Return the estimated number of bytes used by an `H2Matrix`. By default this
includes dense near-field blocks, far-field coupling matrices, and row/column
cluster bases.
"""
function storage_bytes(h::H2Matrix; include_bases::Bool=true)
    bytes = _block_storage_bytes(h)
    if include_bases
        bytes += _basis_storage_bytes(h.row_basis)
        bytes += _basis_storage_bytes(h.col_basis)
    end
    return bytes
end

"""
    dense_storage_bytes(h)

Return the number of bytes required to store the represented matrix densely.
"""
dense_storage_bytes(h::H2Matrix) = prod(size(h)) * sizeof(Float64)

"""
    block_stats(h)

Count H² block-tree nodes and leaves, separated into admissible/uniform and
dense near-field leaves.
"""
function block_stats(h::H2Matrix)
    lvs = leaves(h)
    uniform_leaves = count(l -> l.uniform !== nothing, lvs)
    dense_leaves = count(l -> l.dense !== nothing, lvs)
    empty_leaves = length(lvs) - uniform_leaves - dense_leaves
    return (
        nodes = length(nodes(h)),
        leaves = length(lvs),
        uniform_leaves = uniform_leaves,
        dense_leaves = dense_leaves,
        empty_leaves = empty_leaves,
        depth = depth(h),
    )
end

"""
    rank_stats(h)

Summarize row/column basis ranks and coupling matrix ranks.
"""
function rank_stats(h::H2Matrix)
    row_ranks = [cb.k for cb in nodes(h.row_basis)]
    col_ranks = [cb.k for cb in nodes(h.col_basis)]
    coupling_ranks = Int[]
    for leaf in leaves(h)
        if leaf.uniform !== nothing
            push!(coupling_ranks, min(size(leaf.uniform.S)...))
        end
    end

    summarize(v) = isempty(v) ? (min=0, max=0, mean=0.0, total=0) :
        (min=minimum(v), max=maximum(v), mean=sum(v) / length(v), total=sum(v))

    return (
        row = summarize(row_ranks),
        col = summarize(col_ranks),
        coupling = summarize(coupling_ranks),
    )
end

"""
    compression_summary(h)

Return a compact named tuple with storage, compression, block, and rank
diagnostics.
"""
function compression_summary(h::H2Matrix)
    dense_bytes = dense_storage_bytes(h)
    compressed_bytes = storage_bytes(h)
    return (
        size = size(h),
        dense_bytes = dense_bytes,
        compressed_bytes = compressed_bytes,
        compression_ratio = dense_bytes / compressed_bytes,
        blocks = block_stats(h),
        ranks = rank_stats(h),
    )
end

"""
    dense_reference(K, m, n)

Materialize an indexable matrix-like object into a dense `Matrix{Float64}`.
This is intended for small tests and examples.
"""
function dense_reference(K, m::Integer, n::Integer)
    A = Matrix{Float64}(undef, m, n)
    for j in 1:n, i in 1:m
        A[i, j] = K[i, j]
    end
    return A
end

dense_reference(K, dims::Tuple{Integer,Integer}) = dense_reference(K, dims...)
dense_reference(K::AbstractMatrix) = Matrix{Float64}(K)

"""
    relative_matvec_error(A, B; nsamples=5, rng=Random.GLOBAL_RNG)

Estimate the relative matvec error between two matrix-like operators by applying
them to random vectors and returning the maximum relative error.
"""
function relative_matvec_error(A, B; nsamples::Int=5, rng=Random.GLOBAL_RNG)
    size(A) == size(B) || throw(DimensionMismatch("operator sizes differ"))
    n = size(A, 2)
    worst = 0.0
    for _ in 1:nsamples
        x = randn(rng, n)
        yb = B * x
        denom = norm(yb)
        err = denom == 0 ? norm(A * x - yb) : norm(A * x - yb) / denom
        worst = max(worst, err)
    end
    return worst
end

"""
    sampled_frobenius_error(A, B; nsamples=1024, rng=Random.GLOBAL_RNG)

Estimate relative Frobenius error by sampling entries uniformly. This avoids
forming dense matrices for larger examples.
"""
function sampled_frobenius_error(A, B; nsamples::Int=1024, rng=Random.GLOBAL_RNG)
    size(A) == size(B) || throw(DimensionMismatch("operator sizes differ"))
    m, n = size(A)
    err2 = 0.0
    ref2 = 0.0
    for _ in 1:nsamples
        i = rand(rng, 1:m)
        j = rand(rng, 1:n)
        aij = A[i, j]
        bij = B[i, j]
        err2 += abs2(aij - bij)
        ref2 += abs2(bij)
    end
    return ref2 == 0 ? sqrt(err2) : sqrt(err2 / ref2)
end

function _human_bytes(bytes::Integer)
    bytes < 1024 && return "$(bytes) B"
    bytes < 1024^2 && return "$(round(bytes / 1024; digits=1)) KB"
    bytes < 1024^3 && return "$(round(bytes / 1024^2; digits=1)) MB"
    return "$(round(bytes / 1024^3; digits=2)) GB"
end
