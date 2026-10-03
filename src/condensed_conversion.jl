# ════════════════════════════════════════════════════════════════════
# Condensed (and optionally threaded) H → H² basis construction
# ════════════════════════════════════════════════════════════════════
#
# The adaptive basis of a cluster t is the dominant left singular subspace of
#
#     C_t = [D_t, D_parent[t,:], D_grandparent[t,:], …],
#
# where D_s holds the partner-QR weighted ACA factors of the admissible blocks
# with row (or column) cluster s (see v0.1.1). Left singular vectors and values
# of C_t depend only on the Gram matrix C_t*C_t'. With a thin QR C_t' = Q*R,
# L_t = R' satisfies L_t*L_t' = C_t*C_t', and every row restriction satisfies
# L_t[r,:]*L_t[r,:]' = C_t[r,:]*C_t[r,:]' (Q has orthonormal columns). So each
# child may inherit L_t[child,:] instead of all ancestor blocks, and the transfer
# SVD may use the projection of L_t instead of C_t. Both are exact up to
# rounding, while active widths stay bounded by the cluster size instead of
# growing with depth.
#
# Saturation: when a truncation keeps every coefficient direction (rank equals
# the coefficient dimension), the orthogonal SVD factor is replaced by the
# identity, which spans the same space. A saturated leaf stores V = I and a
# non-leaf stores the identity embedding of its children's coefficients. This is
# an exact change of coordinates; couplings are projected in the same
# coordinates, so the represented operator is unchanged up to rounding.

const _BASIS_GENERAL = 0x00
const _BASIS_IDENTITY = 0x01   # leaf V == I, or non-leaf identity embedding

# Conversion-local node classification (not stored in ClusterBasis, so later
# mutations such as recompression cannot leave stale flags behind).
struct _ConversionBases{N,T}
    index::IdDict{ClusterBasis{N,T},Int}
    kind::Vector{UInt8}
    full::Vector{Bool}     # complete expanded basis is exactly the identity
end

function _ConversionBases(roots::ClusterBasis{N,T}...) where {N,T}
    index = IdDict{ClusterBasis{N,T},Int}()
    for root in roots, cb in nodes(root)
        haskey(index, cb) || (index[cb] = length(index) + 1)
    end
    n = length(index)
    # Vector{Bool} (not BitVector): tasks write distinct elements concurrently.
    return _ConversionBases{N,T}(index, zeros(UInt8, n), fill(false, n))
end

_basis_kind(k::_ConversionBases, cb) = k.kind[k.index[cb]]
_full_identity(k::_ConversionBases, cb) = k.full[k.index[cb]]

struct _BasisBuildContext{D,K}
    data::D
    rtol::Float64
    maxrank::Int
    is_row::Bool
    capped::Vector{Float64}
    lock::ReentrantLock
    kinds::K
    spawn_min::Int   # spawn child subtrees for clusters at least this large
end

function _record_rank_cap!(ctx::_BasisBuildContext, S, k)
    if k == ctx.maxrank && k < length(S) && S[k+1] > ctx.rtol * S[1]
        lock(ctx.lock) do
            push!(ctx.capped, S[k+1] / S[1])
        end
    end
end

_float_matrix(A::Matrix{Float64}) = A
_float_matrix(A) = Matrix{Float64}(A)

_local_rows(child, parent_range) =
    (first(index_range(child.cluster)) - first(parent_range) + 1):(last(index_range(child.cluster)) - first(parent_range) + 1)

# Active matrix C_t = [D_t, inherited] for this cluster, or `nothing` if empty.
# AB' = (A R_B') Q_B': the isometric partner Q drops out, so the weighted
# columns A R_B' measure block error independently of factor scaling. The
# partner QR is the same blocked Householder factorization as `qr(partner)`,
# computed in a per-cluster workspace; TRMM reads only its upper triangle.
function _active_matrix(cb::ClusterBasis, ctx::_BasisBuildContext, inherited)
    m = length(cb)
    entries = get(ctx.data, objectid(cb.cluster), nothing)
    wd = 0; maxn = 0; maxr = 0
    if entries !== nothing
        for b in entries
            partner = ctx.is_row ? b.B : b.A
            n, r = size(partner)
            wd += min(n, r); maxn = max(maxn, n); maxr = max(maxr, r)
        end
    end
    wi = inherited === nothing ? 0 : size(inherited, 2)
    w = wd + wi
    w == 0 && return nothing
    C = Matrix{Float64}(undef, m, w)
    off = 0
    if wd > 0
        work = Matrix{Float64}(undef, maxn, maxr)
        tau = Matrix{Float64}(undef, min(36, maxn, maxr), min(maxn, maxr))
        for b in entries
            partner = ctx.is_row ? b.B : b.A
            factor = ctx.is_row ? b.A : b.B
            n, r = size(partner)
            q = min(n, r)
            q == 0 && continue
            A = view(work, 1:n, 1:r)
            copyto!(A, partner)
            LAPACK.geqrt!(A, view(tau, 1:min(36, q), 1:q))
            dest = view(C, :, (off+1):(off+q))
            if q == r
                copyto!(dest, factor)
                BLAS.trmm!('R', 'U', 'T', 'N', 1.0, view(work, 1:r, 1:r), dest)
            else
                mul!(dest, factor, transpose(triu!(work[1:q, 1:r])))
            end
            off += q
        end
    end
    wi > 0 && copyto!(view(C, :, (off+1):w), inherited)
    return C
end

# Exact width condensation: returns (L, triangular) with L*L' == C*C' (up to
# rounding); L is m × m lower triangular when C is wider than tall.
function _condense_active(C::Matrix{Float64})
    m, w = size(C)
    w <= m && return C, false
    F = qr!(Matrix(transpose(C)))
    return Matrix(transpose(F.R)), true
end

function _leaf_basis!(cb::ClusterBasis, C, ctx::_BasisBuildContext)
    m = length(cb)
    i = ctx.kinds.index[cb]
    if C === nothing
        cb.V = zeros(Float64, m, 0)
        cb.k = 0
        return cb
    end
    F = svd!(C)
    k = _truncation_rank(F.S, ctx.rtol, ctx.maxrank)
    _record_rank_cap!(ctx, F.S, k)
    if k == m
        cb.V = Matrix{Float64}(I, m, m)
        ctx.kinds.kind[i] = _BASIS_IDENTITY
        ctx.kinds.full[i] = true
    else
        cb.V = F.U[:, 1:k]
    end
    cb.k = k
    return cb
end

# out (cb.k × n) = (full basis of cb)' * M, using identity shortcuts.
function _project_into!(out::AbstractMatrix{Float64}, cb::ClusterBasis, M::AbstractMatrix{Float64},
                        kinds::_ConversionBases)
    i = kinds.index[cb]
    if kinds.full[i]
        copyto!(out, M)
    elseif isleaf(cb)
        mul!(out, cb.V', M)
    elseif kinds.kind[i] == _BASIS_IDENTITY
        irange = index_range(cb.cluster)
        off = 0
        for child in cb.children
            child.k == 0 && continue
            _project_into!(view(out, (off+1):(off+child.k), :), child,
                           view(M, _local_rows(child, irange), :), kinds)
            off += child.k
        end
    else
        fill!(out, 0.0)
        cb.k == 0 && return out
        irange = index_range(cb.cluster)
        for child in cb.children
            child.k == 0 && continue
            Mc = view(M, _local_rows(child, irange), :)
            if _full_identity(kinds, child)
                mul!(out, child.E', Mc, 1.0, 1.0)
            else
                tmp = Matrix{Float64}(undef, child.k, size(M, 2))
                _project_into!(tmp, child, Mc, kinds)
                mul!(out, child.E', tmp, 1.0, 1.0)
            end
        end
    end
    return out
end

# (full basis)' * M; returns M itself when the full basis is the identity.
function _project_basis(cb::ClusterBasis, M::Matrix{Float64}, kinds::_ConversionBases)
    _full_identity(kinds, cb) && return M
    return _project_into!(Matrix{Float64}(undef, cb.k, size(M, 2)), cb, M, kinds)
end

# Rigorous sufficient test that a square lower-triangular L keeps every
# direction under truncation: s_min >= 1/||L⁻¹||_F and s_max <= ||L||_F. The
# margin keeps the decision far from the threshold, so it agrees with the
# SVD-based decision of the reference builder (exact up to rounding).
function _certified_full_rank(L::Matrix{Float64}, rtol::Float64)
    nf = norm(L)
    (isfinite(nf) && nf > 0) || return false
    Linv = try
        inv(LowerTriangular(L))
    catch err
        err isa SingularException && return false
        rethrow()
    end
    ni = norm(Linv)
    isfinite(ni) || return false
    return inv(ni * nf) > 16 * max(rtol, size(L, 1) * eps(Float64))
end

function _mark_identity_embedding!(cb::ClusterBasis, kc::Int, kinds::_ConversionBases)
    _set_identity_embedding!(cb, nothing, kc)
    i = kinds.index[cb]
    kinds.kind[i] = _BASIS_IDENTITY
    kinds.full[i] = all(child -> _full_identity(kinds, child), cb.children)
    return cb
end

function _transfer_basis!(cb::ClusterBasis, L, triangular::Bool, ctx::_BasisBuildContext)
    kinds = ctx.kinds
    kc = sum(child.k for child in cb.children)
    if kc == 0
        cb.k = 0
        for child in cb.children
            child.E = zeros(Float64, 0, 0)
        end
        return cb
    elseif L === nothing
        # No direct or ancestor far-field interaction: no parent basis needed.
        _set_empty_parent_basis!(cb)
        return cb
    end
    # With identity children the projected matrix is L itself; a triangular L
    # admits a cheap certificate that no truncation occurs (saturation).
    if triangular && kc <= ctx.maxrank && size(L) == (kc, kc) &&
       all(child -> _full_identity(kinds, child), cb.children) &&
       _certified_full_rank(L, ctx.rtol)
        return _mark_identity_embedding!(cb, kc, kinds)
    end
    irange = index_range(cb.cluster)
    P = Matrix{Float64}(undef, kc, size(L, 2))
    off = 0
    for child in cb.children
        child.k == 0 && continue
        _project_into!(view(P, (off+1):(off+child.k), :), child, view(L, _local_rows(child, irange), :), kinds)
        off += child.k
    end
    F = svd!(P)
    k = _truncation_rank(F.S, ctx.rtol, ctx.maxrank)
    _record_rank_cap!(ctx, F.S, k)
    if k == 0
        _set_empty_parent_basis!(cb)
    elseif k == kc
        # No truncation: the identity embedding spans the same space as F.U.
        _mark_identity_embedding!(cb, kc, kinds)
    else
        cb.k = k
        off = 0
        for child in cb.children
            child.E = F.U[(off+1):(off+child.k), 1:k]
            off += child.k
        end
    end
    return cb
end

function _condensed_basis!(cb::ClusterBasis, inherited, ctx::_BasisBuildContext)
    C = _active_matrix(cb, ctx, inherited)
    inherited = nothing
    isleaf(cb) && return _leaf_basis!(cb, C, ctx)
    L, triangular = C === nothing ? (nothing, false) : _condense_active(C)
    C = nothing
    irange = index_range(cb.cluster)
    # Rows of a lower-triangular L vanish beyond their own index: dropping these
    # zero columns leaves every child's Gram matrix unchanged.
    function sub(child)
        L === nothing && return nothing
        rows = _local_rows(child, irange)
        return triangular ? L[rows, 1:last(rows)] : L[rows, :]
    end
    if length(cb) >= ctx.spawn_min && length(cb.children) > 1
        tasks = [Threads.@spawn(_condensed_basis!(child, sub(child), ctx)) for child in cb.children[2:end]]
        _condensed_basis!(cb.children[1], sub(cb.children[1]), ctx)
        foreach(fetch, tasks)
    else
        for child in cb.children
            _condensed_basis!(child, sub(child), ctx)
        end
    end
    return _transfer_basis!(cb, L, triangular, ctx)
end

_conversion_threads_default() = Threads.nthreads() > 1 && BLAS.get_num_threads() == 1

function _condensed_spawn_min(threads::Bool)
    return threads ? 64 : typemax(Int)
end

"""
Collect admissible leaf block data without copying the ACA factors.
"""
function _collect_rk_data_shared(hmat::HMatrix)
    row_data = Dict{UInt,Vector{NamedTuple{(:A,:B),Tuple{Matrix{Float64},Matrix{Float64}}}}}()
    col_data = Dict{UInt,Vector{NamedTuple{(:A,:B),Tuple{Matrix{Float64},Matrix{Float64}}}}}()
    function visit(h)
        if HMatrices.isleaf(h)
            if HMatrices.isadmissible(h)
                d = HMatrices.data(h)
                if d !== nothing
                    entry = (A=_float_matrix(d.A), B=_float_matrix(d.B))
                    push!(get!(row_data, objectid(HMatrices.rowtree(h)), valtype(row_data)()), entry)
                    push!(get!(col_data, objectid(HMatrices.coltree(h)), valtype(col_data)()), entry)
                end
            end
        else
            foreach(visit, HMatrices.children(h))
        end
    end
    visit(hmat)
    return row_data, col_data
end

function _leaf_pairs!(pairs, h2::H2Matrix, hmat::HMatrix)
    if HMatrices.isleaf(hmat) && isleaf(h2)
        push!(pairs, (h2, hmat))
    elseif !HMatrices.isleaf(hmat) && !isleaf(h2)
        hchildren = HMatrices.children(hmat)
        for j in axes(h2.children, 2), i in axes(h2.children, 1)
            _leaf_pairs!(pairs, h2.children[i, j], hchildren[i, j])
        end
    end
    return pairs
end

# Released blocks are usually old-generation objects that only a full
# collection frees. Consuming builders trigger one after every `threshold`
# released bytes (cheap: few, large objects), so peak memory follows the live
# data instead of growing with uncollected garbage. Numerics are unaffected.
struct _ReleaseTracker
    pending::Threads.Atomic{Int}
    threshold::Int
end
_ReleaseTracker(total::Integer) = _ReleaseTracker(Threads.Atomic{Int}(0), max(32 * 2^20, Int(total) ÷ 8))
_released!(::Nothing, bytes) = nothing
function _released!(t::_ReleaseTracker, bytes::Integer)
    p = Threads.atomic_add!(t.pending, Int(bytes)) + Int(bytes)
    if p >= t.threshold && Threads.atomic_cas!(t.pending, p, 0) == p
        GC.gc(true)
    end
    return nothing
end

_hblock_bytes(d::Nothing) = 0
_hblock_bytes(d::AbstractMatrix) = d isa HMatrices.RkMatrix ? sizeof(d.A) + sizeof(d.B) : sizeof(d)

function _fill_leaf!(h2::H2Matrix, hmat::HMatrix, kinds::_ConversionBases, consume::Bool,
                     tracker=nothing)
    rb = h2.row_basis
    cb = h2.col_basis
    d = HMatrices.data(hmat)
    if HMatrices.isadmissible(hmat)
        S = if d === nothing || rb.k == 0 || cb.k == 0
            zeros(Float64, rb.k, cb.k)
        else
            VA = _project_basis(rb, _float_matrix(d.A), kinds)
            VB = _project_basis(cb, _float_matrix(d.B), kinds)
            VA * VB'
        end
        h2.uniform = UniformBlock(rb, cb, S)
    else
        h2.dense = if d === nothing
            zeros(Float64, length(rb), length(cb))
        elseif consume
            _float_matrix(d)     # ownership moves from the consumed H-matrix
        else
            Matrix{Float64}(d)
        end
    end
    if consume && d !== nothing
        HMatrices.setdata!(hmat, nothing)
        # Near-field blocks moved into h2 without a copy release nothing.
        HMatrices.isadmissible(hmat) && _released!(tracker, _hblock_bytes(d))
    end
    return nothing
end

function _fill_h2_condensed!(h2::H2Matrix, hmat::HMatrix, kinds::_ConversionBases;
                             threads::Bool, consume::Bool)
    pairs = _leaf_pairs!(Tuple{typeof(h2),typeof(hmat)}[], h2, hmat)
    tracker = consume ?
        _ReleaseTracker(sum((_hblock_bytes(HMatrices.data(hm)) for (_, hm) in pairs); init=0)) : nothing
    ntasks = threads ? min(Threads.nthreads(), length(pairs)) : 1
    if ntasks <= 1
        for (h, hm) in pairs
            _fill_leaf!(h, hm, kinds, consume, tracker)
        end
    else
        # Largest blocks first for load balance; each block is independent, so
        # the result does not depend on scheduling.
        order = sortperm([length(h.row_basis) * length(h.col_basis) for (h, _) in pairs]; rev=true)
        next = Threads.Atomic{Int}(1)
        tasks = map(1:ntasks) do _
            Threads.@spawn begin
                while true
                    n = Threads.atomic_add!(next, 1)
                    n > length(order) && break
                    h, hm = pairs[order[n]]
                    _fill_leaf!(h, hm, kinds, consume, tracker)
                end
            end
        end
        foreach(fetch, tasks)
    end
    return h2
end

function _compress_hmatrix_to_h2_condensed(hmat::HMatrix; rtol, maxrank, strict, threads, consume)
    rt = HMatrices.rowtree(hmat)
    ct = HMatrices.coltree(hmat)
    rb = build_cluster_basis(rt)
    cb = build_cluster_basis(ct)
    row_map = _build_obj_map(rb)
    col_map = _build_obj_map(cb)
    kinds = _ConversionBases(rb, cb)
    row_data, col_data = _collect_rk_data_shared(hmat)
    capped = Float64[]
    lk = ReentrantLock()
    spawn_min = _condensed_spawn_min(threads)
    rctx = _BasisBuildContext(row_data, rtol, maxrank, true, capped, lk, kinds, spawn_min)
    cctx = _BasisBuildContext(col_data, rtol, maxrank, false, capped, lk, kinds, spawn_min)
    if threads
        # Row and column bases are independent.
        task = Threads.@spawn _condensed_basis!(cb, nothing, cctx)
        _condensed_basis!(rb, nothing, rctx)
        fetch(task)
    else
        _condensed_basis!(rb, nothing, rctx)
        _condensed_basis!(cb, nothing, cctx)
    end
    # Bases are complete: drop the factor references so a consuming fill can
    # release each ACA block as soon as its coupling is formed.
    empty!(row_data); empty!(col_data)
    if !isempty(capped)
        message = "H2 basis rank cap prevents the requested local tolerance at $(length(capped)) clusters; largest relative discarded singular value = $(maximum(capped)). Increase maxrank."
        strict ? throw(ArgumentError(message)) : (@warn message)
    end
    h2 = _mirror_hmat_to_h2(hmat, row_map, col_map)
    _fill_h2_condensed!(h2, hmat, kinds; threads, consume)
    return h2
end
