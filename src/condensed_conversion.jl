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

const _RkEntry = NamedTuple{(:A,:B),Tuple{Matrix{Float64},Matrix{Float64}}}
const _NO_ENTRIES = _RkEntry[]

struct _BasisBuildContext{K,F}
    # Admissible factor pairs by node index (`kinds.index`): blocks whose row
    # (for row bases) or column (for column bases) cluster is that node. Each
    # task only touches its own node's slot, which is emptied once used.
    data::Vector{Vector{_RkEntry}}
    rtol::Float64
    maxrank::Int
    is_row::Bool
    capped::Vector{Float64}
    lock::ReentrantLock
    kinds::K
    spawn_min::Int   # spawn child subtrees for clusters at least this large
    fill::F          # `nothing` or an `_EagerFill` run as nodes become final
end

# Node-indexed factor lists from a Dict keyed by `objectid(cluster)`.
function _node_entries(data::AbstractDict, root::ClusterBasis, kinds::_ConversionBases)
    out = [_NO_ENTRIES for _ in 1:length(kinds.kind)]
    for cb in nodes(root)
        e = get(data, objectid(cb.cluster), nothing)
        e === nothing || (out[kinds.index[cb]] = collect(_RkEntry, e))
    end
    return out
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

# The builder works with transposed active sets Cᵗ = C_t' (rows = columns of
# C_t), so the condensing QR runs in place and children read their inherited
# factor as a column block of the parent's R (= L_t') without copies.
#
# Cᵗ = [D_t'; inherited'] for this cluster, or `nothing` if empty.
# AB' = (A R_B') Q_B': the isometric partner Q drops out, so the weighted
# columns A R_B' measure block error independently of factor scaling. The
# partner QR is the same blocked Householder factorization as `qr(partner)`,
# computed in a per-cluster workspace; TRMM reads only its upper triangle.
function _active_matrix_t(cb::ClusterBasis, ctx::_BasisBuildContext, inherited_t)
    m = length(cb)
    slot = ctx.kinds.index[cb]
    entries = ctx.data[slot]
    # Drop this cluster's factor references once its active set is formed, so
    # consumed H blocks can be released as soon as their couplings exist.
    ctx.data[slot] = _NO_ENTRIES
    isempty(entries) && (entries = nothing)
    wd = 0; maxn = 0; maxr = 0
    if entries !== nothing
        for b in entries
            partner = ctx.is_row ? b.B : b.A
            n, r = size(partner)
            wd += min(n, r); maxn = max(maxn, n); maxr = max(maxr, r)
        end
    end
    wi = inherited_t === nothing ? 0 : size(inherited_t, 1)
    w = wd + wi
    w == 0 && return nothing
    Ct = Matrix{Float64}(undef, w, m)
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
            dest = view(Ct, (off+1):(off+q), :)
            if q == r
                transpose!(dest, factor)
                BLAS.trmm!('L', 'U', 'N', 'N', 1.0, view(work, 1:r, 1:r), dest)
            else
                mul!(dest, triu!(work[1:q, 1:r]), transpose(factor))
            end
            off += q
        end
    end
    wi > 0 && copyto!(view(Ct, (off+1):w, :), inherited_t)
    return Ct
end

# Exact width condensation in place: returns (Lt, triangular) with
# Lt'*Lt == Ct'*Ct (up to rounding). For Ct taller than wide, Lt is the m × m
# upper-triangular R of Ct = Q*R (a view into Ct, lower part zeroed).
function _condense_active_t!(Ct::Matrix{Float64})
    w, m = size(Ct)
    w <= m && return Ct, false
    LAPACK.geqrt!(Ct, Matrix{Float64}(undef, min(36, m), m))
    R = view(Ct, 1:m, 1:m)
    for j in 1:m, i in (j+1):m
        R[i, j] = 0.0
    end
    return R, true
end

function _leaf_basis!(cb::ClusterBasis, Ct, ctx::_BasisBuildContext)
    m = length(cb)
    i = ctx.kinds.index[cb]
    if Ct === nothing
        cb.V = zeros(Float64, m, 0)
        cb.k = 0
        return cb
    end
    # Left singular vectors of C_t are the right singular vectors of Ct.
    F = svd!(Ct)
    k = _truncation_rank(F.S, ctx.rtol, ctx.maxrank)
    _record_rank_cap!(ctx, F.S, k)
    if k == m
        cb.V = Matrix{Float64}(I, m, m)
        ctx.kinds.kind[i] = _BASIS_IDENTITY
        ctx.kinds.full[i] = true
    else
        cb.V = Matrix(transpose(view(F.Vt, 1:k, :)))
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

# out (n × cb.k) = M * (full basis of cb), M with |cb| columns; the
# transposed counterpart of `_project_into!`.
function _rproject_into!(out::AbstractMatrix{Float64}, cb::ClusterBasis, M::AbstractMatrix{Float64},
                         kinds::_ConversionBases)
    i = kinds.index[cb]
    if kinds.full[i]
        copyto!(out, M)
    elseif isleaf(cb)
        mul!(out, M, cb.V)
    elseif kinds.kind[i] == _BASIS_IDENTITY
        irange = index_range(cb.cluster)
        off = 0
        for child in cb.children
            child.k == 0 && continue
            _rproject_into!(view(out, :, (off+1):(off+child.k)), child,
                            view(M, :, _local_rows(child, irange)), kinds)
            off += child.k
        end
    else
        fill!(out, 0.0)
        cb.k == 0 && return out
        irange = index_range(cb.cluster)
        for child in cb.children
            child.k == 0 && continue
            Mc = view(M, :, _local_rows(child, irange))
            if _full_identity(kinds, child)
                mul!(out, Mc, child.E, 1.0, 1.0)
            else
                tmp = Matrix{Float64}(undef, size(M, 1), child.k)
                _rproject_into!(tmp, child, Mc, kinds)
                mul!(out, tmp, child.E, 1.0, 1.0)
            end
        end
    end
    return out
end

# Rigorous sufficient test that a square upper-triangular R keeps every
# direction under truncation: s_min >= 1/||R⁻¹||_F and s_max <= ||R||_F. The
# margin keeps the decision far from the threshold, so it agrees with the
# SVD-based decision of the reference builder (exact up to rounding).
function _certified_full_rank(R::AbstractMatrix{Float64}, rtol::Float64)
    nf = norm(R)
    (isfinite(nf) && nf > 0) || return false
    Rinv = try
        inv(UpperTriangular(R))
    catch err
        err isa SingularException && return false
        rethrow()
    end
    ni = norm(Rinv)
    isfinite(ni) || return false
    return inv(ni * nf) > 16 * max(rtol, size(R, 1) * eps(Float64))
end

# Identity-embedding transfers are dense selection matrices that the builder
# and coupling fill never read (identity nodes are projected by stacking), so
# they are materialized only at the end, after consumed H blocks are released.
const _DEFERRED_TRANSFER = zeros(Float64, 0, 0)

# R of Pt = Q*R when Pt is taller than wide (same singular values and right
# singular vectors), otherwise Pt itself.
function _square_factor!(Pt::Matrix{Float64})
    w, n = size(Pt)
    w <= n && return Pt
    LAPACK.geqrt!(Pt, Matrix{Float64}(undef, min(36, n), n))
    R = view(Pt, 1:n, 1:n)    # in place: zero the Householder vectors below R
    for j in 1:n, i in (j+1):n
        R[i, j] = 0.0
    end
    return R
end

function _mark_identity_embedding!(cb::ClusterBasis, kc::Int, kinds::_ConversionBases)
    cb.k = kc
    for child in cb.children
        child.E = _DEFERRED_TRANSFER
    end
    i = kinds.index[cb]
    kinds.kind[i] = _BASIS_IDENTITY
    kinds.full[i] = all(child -> _full_identity(kinds, child), cb.children)
    return cb
end

function _transfer_basis!(cb::ClusterBasis, Lt, triangular::Bool, ctx::_BasisBuildContext)
    kinds = ctx.kinds
    kc = sum(child.k for child in cb.children)
    if kc == 0
        cb.k = 0
        for child in cb.children
            child.E = zeros(Float64, 0, 0)
        end
        return cb
    elseif Lt === nothing
        # No direct or ancestor far-field interaction: no parent basis needed.
        _set_empty_parent_basis!(cb)
        return cb
    end
    # With identity children the projected matrix is Lt itself; a triangular
    # Lt admits a cheap certificate that no truncation occurs (saturation).
    if triangular && kc <= ctx.maxrank && size(Lt) == (kc, kc) &&
       all(child -> _full_identity(kinds, child), cb.children) &&
       _certified_full_rank(Lt, ctx.rtol)
        return _mark_identity_embedding!(cb, kc, kinds)
    end
    irange = index_range(cb.cluster)
    Pt = Matrix{Float64}(undef, size(Lt, 1), kc)
    off = 0
    for child in cb.children
        child.k == 0 && continue
        _rproject_into!(view(Pt, :, (off+1):(off+child.k)), child, view(Lt, :, _local_rows(child, irange)), kinds)
        off += child.k
    end
    # Left singular vectors of the projected active set = right ones of Pt,
    # which a tall Pt shares with its triangular QR factor (cheaper SVD).
    F = svd!(_square_factor!(Pt))
    k = _truncation_rank(F.S, ctx.rtol, ctx.maxrank)
    _record_rank_cap!(ctx, F.S, k)
    if k == 0
        _set_empty_parent_basis!(cb)
    elseif k == kc
        # No truncation: the identity embedding spans the same space.
        _mark_identity_embedding!(cb, kc, kinds)
    else
        cb.k = k
        off = 0
        for child in cb.children
            child.E = Matrix(transpose(view(F.Vt, 1:k, (off+1):(off+child.k))))
            off += child.k
        end
    end
    return cb
end

function _condensed_basis!(cb::ClusterBasis, inherited_t, ctx::_BasisBuildContext)
    Ct = _active_matrix_t(cb, ctx, inherited_t)
    inherited_t = nothing
    if isleaf(cb)
        _leaf_basis!(cb, Ct, ctx)
        return _node_ready!(ctx.fill, cb)
    end
    Lt, triangular = Ct === nothing ? (nothing, false) : _condense_active_t!(Ct)
    irange = index_range(cb.cluster)
    # Children inherit column blocks of Lt (views; Lt stays alive until they
    # finish). Columns of an upper-triangular R vanish below their own index:
    # dropping these zero rows leaves every child's Gram matrix unchanged.
    function sub(child)
        Lt === nothing && return nothing
        cols = _local_rows(child, irange)
        return triangular ? view(Lt, 1:last(cols), cols) : view(Lt, :, cols)
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
    _transfer_basis!(cb, Lt, triangular, ctx)
    return _node_ready!(ctx.fill, cb)
end

function _materialize_identity_embeddings!(root::ClusterBasis, kinds::_ConversionBases)
    for cb in nodes(root)
        if !isleaf(cb) && kinds.kind[kinds.index[cb]] == _BASIS_IDENTITY
            _set_identity_embedding!(cb, nothing, cb.k)
        end
    end
    return root
end

_conversion_threads_default() = Threads.nthreads() > 1 && BLAS.get_num_threads() == 1

function _condensed_spawn_min(threads::Bool)
    return threads ? 64 : typemax(Int)
end

"""
Collect admissible leaf block data by row and column basis node, without
copying the ACA factors.
"""
function _collect_rk_entries(pairs, kinds::_ConversionBases)
    data = [_RkEntry[] for _ in 1:length(kinds.kind)]
    for (h, hm) in pairs
        HMatrices.isadmissible(hm) || continue
        d = HMatrices.data(hm)
        d === nothing && continue
        entry = (A=_float_matrix(d.A), B=_float_matrix(d.B))
        push!(data[kinds.index[h.row_basis]], entry)
        push!(data[kinds.index[h.col_basis]], entry)
    end
    return data
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

# A coupling needs only the final bases of its row and column clusters, which
# are final once each node's own transfer step is done. Each admissible block
# is therefore filled (and, when consuming, its ACA factors released) by the
# task that completes the second of its two nodes, overlapping the fill with
# the remaining basis work. The result does not depend on the schedule.
struct _EagerFill{P,K,T}
    pairs::Vector{P}
    lists::Vector{Vector{Int}}           # by node index: pairs of that node
    pending::Vector{Threads.Atomic{Int}} # nodes still to finish per pair
    kinds::K
    consume::Bool
    tracker::T
end

function _EagerFill(pairs::Vector{P}, kinds::_ConversionBases, consume::Bool, tracker) where {P}
    lists = [Int[] for _ in 1:length(kinds.kind)]
    for (p, (h, _)) in enumerate(pairs)
        push!(lists[kinds.index[h.row_basis]], p)
        push!(lists[kinds.index[h.col_basis]], p)
    end
    pending = [Threads.Atomic{Int}(2) for _ in pairs]
    return _EagerFill(pairs, lists, pending, kinds, consume, tracker)
end

_node_ready!(::Nothing, cb) = cb
function _node_ready!(f::_EagerFill, cb::ClusterBasis)
    for p in f.lists[f.kinds.index[cb]]
        if Threads.atomic_sub!(f.pending[p], 1) == 1
            h, hm = f.pairs[p]
            _fill_leaf!(h, hm, f.kinds, f.consume, f.tracker)
        end
    end
    return cb
end

function _compress_hmatrix_to_h2_condensed(hmat::HMatrix; rtol, maxrank, strict, threads, consume)
    rt = HMatrices.rowtree(hmat)
    ct = HMatrices.coltree(hmat)
    rb = build_cluster_basis(rt)
    cb = build_cluster_basis(ct)
    row_map = _build_obj_map(rb)
    col_map = _build_obj_map(cb)
    kinds = _ConversionBases(rb, cb)
    h2 = _mirror_hmat_to_h2(hmat, row_map, col_map)
    pairs = _leaf_pairs!(Tuple{typeof(h2),typeof(hmat)}[], h2, hmat)
    admissible = filter(p -> HMatrices.isadmissible(p[2]), pairs)
    data = _collect_rk_entries(admissible, kinds)
    tracker = consume ?
        _ReleaseTracker(sum((_hblock_bytes(HMatrices.data(hm)) for (_, hm) in admissible); init=0)) : nothing
    fill = _EagerFill(admissible, kinds, consume, tracker)
    capped = Float64[]
    lk = ReentrantLock()
    spawn_min = _condensed_spawn_min(threads)
    rctx = _BasisBuildContext(data, rtol, maxrank, true, capped, lk, kinds, spawn_min, fill)
    cctx = _BasisBuildContext(data, rtol, maxrank, false, capped, lk, kinds, spawn_min, fill)
    if threads
        # Row and column bases are independent.
        task = Threads.@spawn _condensed_basis!(cb, nothing, cctx)
        _condensed_basis!(rb, nothing, rctx)
        fetch(task)
    else
        _condensed_basis!(rb, nothing, rctx)
        _condensed_basis!(cb, nothing, cctx)
    end
    if !isempty(capped)
        message = "H2 basis rank cap prevents the requested local tolerance at $(length(capped)) clusters; largest relative discarded singular value = $(maximum(capped)). Increase maxrank."
        strict ? throw(ArgumentError(message)) : (@warn message)
    end
    for (h, hm) in pairs
        HMatrices.isadmissible(hm) || _fill_leaf!(h, hm, kinds, consume, tracker)
    end
    _materialize_identity_embeddings!(rb, kinds)
    _materialize_identity_embeddings!(cb, kinds)
    return h2
end
