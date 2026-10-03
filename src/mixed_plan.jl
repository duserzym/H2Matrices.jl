# Adaptive mixed-precision packet plan.
#
# Each far-field packet P_t (all couplings of one row node, as in
# H2PacketMatvecPlan) is rotated to the left singular basis of P_t,
# P_t = U_t W_t, so that row i of W_t has norm omega_i ~ sigma_i(P_t).
# Rows with small omega_i are stored in Float32 and all products accumulate in
# Float64. Rounding row i to Float32 perturbs the operator by at most
# u32*omega_i in the Frobenius norm, so precision follows singular-value
# weight; the leading rows stay Float64.
#
# Rows of one packet are stored in an interleaved layout (groups of four rows,
# column-interleaved), so each packet is one sequential memory stream. The
# kernels read it with SIMD and accumulate in Float64.

const _U32 = Float64(eps(Float32)) / 2      # unit roundoff of Float32 rounding to nearest

const _U48 = 2.0^-37                        # unit roundoff of the 48-bit format below

# Optional 48-bit format: a Float64 truncated to sign, exponent and 36 mantissa
# bits (round to nearest), stored as a UInt32 plane and a UInt16 plane.
struct _T48 end
struct _Planes48
    hi::Vector{UInt32}
    lo::Vector{UInt16}
end
_storage(::Type{T}, n) where {T<:Union{Float32,Float64}} = Vector{T}(undef, n)
_storage(::Type{_T48}, n) = _Planes48(Vector{UInt32}(undef, n), Vector{UInt16}(undef, n))
Base.@propagate_inbounds _put!(d::Vector{T}, q, v) where {T} = (d[q] = T(v); nothing)
Base.@propagate_inbounds function _put!(d::_Planes48, q, v)
    u = (reinterpret(UInt64, Float64(v)) + 0x0000_0000_0000_8000) & 0xffff_ffff_ffff_0000
    d.hi[q] = UInt32(u >> 32); d.lo[q] = UInt16((u >> 16) & 0xffff)
    nothing
end
Base.@propagate_inbounds _ld(d::Vector{T}, q) where {T} = Float64(d[q])
Base.@propagate_inbounds _ld(d::_Planes48, q) = reinterpret(Float64, (UInt64(d.hi[q]) << 32) | (UInt64(d.lo[q]) << 16))
_nbytes(d::Vector) = sizeof(d)
_nbytes(d::_Planes48) = sizeof(d.hi) + sizeof(d.lo)

# Row-interleaved storage of a dense `rows x N` matrix: groups of four rows are
# stored column-interleaved (`data[4N*g+4(j-1)+q] = A[4g+q,j]`), and the trailing
# `rows % 4` rows follow, each contiguous.
struct _InterleavedRows{T,D}
    data::D
    rows::Int
    N::Int
end
const _Rows64 = _InterleavedRows{Float64,Vector{Float64}}
const _Rows48 = _InterleavedRows{_T48,_Planes48}
const _Rows32 = _InterleavedRows{Float32,Vector{Float32}}
# Position of entry (i, j) in the interleaved storage of a `rows x N` matrix.
@inline function _il_index(rows, N, i, j)
    g4 = rows ÷ 4; g = (i - 1) >> 2
    g < g4 ? 4N * g + 4(j - 1) + (i - 4g) : 4N * g4 + N * (i - 4g4 - 1) + j
end
_InterleavedRows{T}(::UndefInitializer, rows::Int, N::Int) where {T} =
    (d = _storage(T, rows * N); _InterleavedRows{T,typeof(d)}(d, rows, N))
# Store A (rows x n) into columns coloff+1:coloff+n.
function _put_block!(S::_InterleavedRows, A::AbstractMatrix, coloff::Int)
    rows, N, d = S.rows, S.N, S.data
    @inbounds for j in axes(A, 2), i in 1:rows
        _put!(d, _il_index(rows, N, i, coloff + j), A[i, j])
    end
    S
end
# Squared Frobenius norm of (stored - A) over columns coloff+1:coloff+n, without copies.
function _rounding_err2(S::_InterleavedRows, A::AbstractMatrix, coloff::Int=0)
    rows, N, d = S.rows, S.N, S.data; e = 0.0
    @inbounds for j in axes(A, 2), i in 1:rows
        e += abs2(_ld(d, _il_index(rows, N, i, coloff + j)) - A[i, j])
    end
    e
end
_InterleavedRows{T}(A::AbstractMatrix) where {T} = _put_block!(_InterleavedRows{T}(undef, size(A)...), A, 0)
_eltype(::_InterleavedRows{T}) where {T} = T === _T48 ? Float64 : T
function Base.Matrix(S::_InterleavedRows)
    M = Matrix{_eltype(S)}(undef, S.rows, S.N)
    for j in 1:S.N, i in 1:S.rows
        M[i, j] = _ld(S.data, _il_index(S.rows, S.N, i, j))
    end
    M
end
@inline function _il_fwd4!(z, zo, d, o, N, x)
    s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0
    @inbounds @simd for j in 1:N
        xj = x[j]; q = o + 4(j - 1)
        s1 = muladd(_ld(d, q+1), xj, s1); s2 = muladd(_ld(d, q+2), xj, s2)
        s3 = muladd(_ld(d, q+3), xj, s3); s4 = muladd(_ld(d, q+4), xj, s4)
    end
    @inbounds begin
        z[zo+1] += s1; z[zo+2] += s2; z[zo+3] += s3; z[zo+4] += s4
    end
    nothing
end
@inline function _il_fwd1!(z, zo, d, o, N, x)
    s = 0.0
    @inbounds @simd for j in 1:N
        s = muladd(_ld(d, o+j), x[j], s)
    end
    @inbounds z[zo+1] += s
    nothing
end
# z[zo+i] += sum_j A[i,j] * x[j]
function _il_forward!(z::Vector{Float64}, zo::Int, A::_InterleavedRows, x::Vector{Float64})
    N = A.N; g4 = A.rows ÷ 4; d = A.data
    for g in 0:g4-1
        _il_fwd4!(z, zo + 4g, d, 4N * g, N, x)
    end
    for r in 1:A.rows%4
        _il_fwd1!(z, zo + 4g4 + r - 1, d, 4N * g4 + N * (r - 1), N, x)
    end
    nothing
end
@inline function _il_adj4!(x, d, o, N, w, wo)
    @inbounds w1 = w[wo+1]; @inbounds w2 = w[wo+2]; @inbounds w3 = w[wo+3]; @inbounds w4 = w[wo+4]
    @inbounds @simd for j in 1:N
        q = o + 4(j - 1)
        x[j] = muladd(_ld(d, q+4), w4, muladd(_ld(d, q+3), w3, muladd(_ld(d, q+2), w2, muladd(_ld(d, q+1), w1, x[j]))))
    end
    nothing
end
@inline function _il_adj1!(x, d, o, N, wi)
    @inbounds @simd for j in 1:N
        x[j] = muladd(_ld(d, o+j), wi, x[j])
    end
    nothing
end
# x[j] += sum_i A[i,j] * w[wo+i]   (exact transpose of _il_forward! on the same stored data)
function _il_adjoint!(x::Vector{Float64}, A::_InterleavedRows, w::Vector{Float64}, wo::Int)
    N = A.N; g4 = A.rows ÷ 4; d = A.data
    for g in 0:g4-1
        _il_adj4!(x, d, 4N * g, N, w, wo + 4g)
    end
    for r in 1:A.rows%4
        @inbounds wi = w[wo+4g4+r]
        _il_adj1!(x, d, 4N * g4 + N * (r - 1), N, wi)
    end
    nothing
end

struct _MixedPacket
    row::UnitRange{Int}
    columns::Vector{UnitRange{Int}}
    hi::_Rows64
    mid::_Rows48
    lo::_Rows32
    # Explicit packet rotation Q = H_1 ⋯ H_r as Householder reflectors (empty if
    # none or absorbed into the row basis): reflector i is [1; hv[off_i+1:off_i+k-i]].
    hv::Vector{Float64}
    tau::Vector{Float64}
    scratch::Vector{Float64}
    zbuf::Vector{Float64}
end
_copy_packet(b::_MixedPacket) = _MixedPacket(b.row, b.columns, b.hi, b.mid, b.lo, b.hv, b.tau, zeros(length(b.scratch)), zeros(length(b.zbuf)))
_packet_bytes(b::_MixedPacket) = _nbytes(b.hi.data) + _nbytes(b.mid.data) + _nbytes(b.lo.data) + sizeof(b.hv) + sizeof(b.tau)
# Stored Float64 entries of r Householder reflectors of a k-row rotation.
_reflector_entries(k, r) = r * k - (r * (r + 1)) ÷ 2
@inline function _reflect!(z, hv, off, i, k, τ)
    m = k - i; s = @inbounds z[i]
    @inbounds @simd for l in 1:m
        s = muladd(hv[off+l], z[i+l], s)
    end
    s *= τ
    @inbounds z[i] -= s
    @inbounds @simd for l in 1:m
        z[i+l] = muladd(-s, hv[off+l], z[i+l])
    end
    nothing
end
# z <- Q z (transposed=false) or z <- Q' z, Q = H_1 ⋯ H_r; each H_i is symmetric,
# so both directions apply exactly the same stored reflectors.
function _apply_reflectors!(z::Vector{Float64}, hv::Vector{Float64}, tau::Vector{Float64}, transposed::Bool)
    k = length(z); r = length(tau)
    if transposed
        off = 0
        for i in 1:r
            _reflect!(z, hv, off, i, k, tau[i]); off += k - i
        end
    else
        for i in r:-1:1
            _reflect!(z, hv, (i - 1) * k - ((i - 1) * i) ÷ 2, i, k, tau[i])
        end
    end
    nothing
end

"""
    H2MixedPacketMatvecPlan(compact_plan; workers=1, precision_rtol=1e-13, format48=false)
    H2MixedPacketMatvecPlan(h2; workers=1, precision_rtol=1e-13, format48=false)

Packet plan with adaptive mixed-precision storage. Far-field packets hold all
couplings of one row node, as in `H2PacketMatvecPlan`. Near-field packets hold
all dense blocks of one leaf row range.

Each packet `P` (`k × N`) is rotated to its left singular basis, `P = Q W`.
Row `i` of `W` then has norm `ω_i = σ_i(P)`. Trailing rows with small weight are
stored in Float32 and the leading rows in Float64. Every product accumulates
in Float64; Float32 values are only widened, never used as accumulators.
Rotations of explicit (unsaturated) row bases are absorbed exactly into the
stored transfers and leaf bases, so they cost no storage. For implicit
(saturated) or physical rows, `Q` is stored as the `r` Householder reflectors
spanning the rows above the lowest-precision block (about `8r(k - r/2)` bytes).
That cost enters the selection. The rounding cost depends only on these
subspaces, not on the basis chosen inside them.

With `format48=true`, rows of intermediate weight can also use a 48-bit format:
a Float64 rounded to 36 mantissa bits (unit roundoff `u₄₈ = 2⁻³⁷`), stored as
a UInt32 plane plus a UInt16 plane. This lowers storage further, but its
decode makes those rows slower to stream than Float64 rows on CPUs where a
few workers already saturate memory bandwidth.

A global Lagrangian allocation picks the rows. It minimizes stored bytes
subject to the a priori bound

    ‖Ã - A‖₂ ≤ ‖Ã - A‖_F ≤ (Σ_t β_t² [Σ_{i∈f32(t)} (u₃₂² ω_i² + N_t 2⁻³⁰⁰) + Σ_{i∈f48(t)} u₄₈² ω_i²])^{1/2} ≤ precision_rtol · η.

Here `A` is the Float64 compact operator and `Ã` the stored mixed operator,
both in exact arithmetic, and `u₃₂ = 2⁻²⁴`. `β_t` is the product of the row-
and column-basis 2-norms (1 for orthonormal, implicit or physical bases).
`η = (Σ ‖stored block‖_F²)^{1/2}` equals `‖A‖_F` for orthonormal bases.
The bound is additive over packets because packets cover disjoint matrix
blocks. The `2⁻³⁰⁰` terms cover Float32 underflow. For Gaussian inputs the
bound limits the root-mean-square relative perturbation of products:
`E‖(Ã-A)x‖² / E‖Ax‖² ≤ precision_rtol²`. `precision_rtol=0` keeps everything
in Float64 and differs from `H2PacketMatvecPlan` only by summation order.

Forward and adjoint products apply the same stored values, so
`adjoint(plan)` is the exact transpose of the stored mixed operator. With
`workers>1`, far and near packets run in one parallel phase under a balanced
static schedule, with private transpose reduction buffers. Use BLAS
threads=1. `copy(plan)` shares the numerical data and gives private scratch.
`precision_summary(plan)` reports the selection and both bounds.
"""
struct H2MixedPacketMatvecPlan <: AbstractMatrix{Float64}
    shape::Tuple{Int,Int}
    rows::Vector{_CompactBasisNode}
    cols::Vector{_CompactBasisNode}
    packets::Vector{_MixedPacket}
    nearpackets::Vector{_MixedPacket}
    farschedule::Vector{Vector{Int}}
    nearschedule::Vector{Vector{Int}}
    rowcoeff::Vector{Float64}
    colcoeff::Vector{Float64}
    rowbuffer::Vector{Float64}
    colbuffer::Vector{Float64}
    rowperm::Vector{Int}
    colperm::Vector{Int}
    partials::Vector{Vector{Float64}}
    nearpartials::Vector{Vector{Float64}}
    precision::NamedTuple
end
H2MixedPacketMatvecPlan(h::H2Matrix; kwargs...) = H2MixedPacketMatvecPlan(H2CompactMatvecPlan(h); kwargs...)

# Spectral norm of each compact basis node's full (nested) basis, via Gram matrices.
function _compact_basis_norms(nodes::Vector{_CompactBasisNode})
    G = Vector{Matrix{Float64}}(undef, length(nodes)); nrm = ones(length(nodes))
    for i in reverse(eachindex(nodes))
        n = nodes[i]; k = length(n.coeff)
        if n.identity || k == 0
            G[i] = zeros(0, 0)
            continue
        elseif isempty(n.children)
            G[i] = n.V' * n.V
        else
            g = zeros(k, k)
            for j in n.children
                c = nodes[j]
                (isempty(c.coeff) || isempty(c.E)) && continue
                g .+= c.identity ? c.E' * c.E : c.E' * G[j] * c.E
            end
            G[i] = g
        end
        # Divide and conquer avoids MRRR (syevr) failures on graded Gram matrices.
        nrm[i] = sqrt(max(maximum(eigvals(Symmetric(G[i]); alg=LinearAlgebra.DivideAndConquer()); init=0.0), 0.0))
    end
    nrm
end

function _lpt_schedule(costs::Vector{Float64}, workers::Int)
    owner = zeros(Int, length(costs)); load = zeros(workers)
    for i in sortperm(costs; rev=true)
        w = argmin(load); owner[i] = w; load[w] += costs[i]
    end
    owner
end

# Rows sorted by decreasing weight are stored as Float64 rows 1:m₄₈, optional
# 48-bit rows m₄₈+1:m and Float32 rows m+1:k. For multiplier mu this returns the
# split (r32 = k-m, r48 = m-m₄₈) maximizing saved bytes - mu*cost and its cost.
# An explicit rotation stores the reflectors spanning the rows above the last
# format boundary; any orthonormal basis of the remaining complement gives the
# same rounding cost, because the cost depends only on the subspaces.
function _best_split(om2::Vector{Float64}, N::Int, explicit::Bool, beta2::Float64, mu::Float64, use48::Bool)
    k = length(om2)
    m48 = use48 ? count(w -> 2.0 * N <= mu * beta2 * _U48^2 * w, om2) : k   # rows that prefer Float64 over 48-bit
    p48 = zeros(k + 1)
    for i in 1:k
        p48[i+1] = p48[i] + beta2 * _U48^2 * om2[i]
    end
    best = 0.0; bestr = (0, 0); bestc = 0.0; c32 = 0.0
    for r32 in 0:k
        r32 > 0 && (c32 += beta2 * (_U32^2 * om2[k-r32+1] + N * 2.0^-300))
        m = k - r32; mm = min(m48, m); n48 = m - mm
        (r32 == 0 && n48 == 0) && continue
        nref = r32 > 0 ? m : (mm < k ? mm : 0)
        c = c32 + p48[m+1] - p48[mm+1]
        f = 4.0 * r32 * N + 2.0 * n48 * N - (explicit ? 8.0 * (_reflector_entries(k, nref) + nref) : 0.0) - mu * c
        if f > best
            best = f; bestr = (r32, n48); bestc = c
        end
    end
    bestr, bestc
end
# Smallest Lagrange multiplier whose per-packet selections meet the squared budget.
function _precision_selection(om2, Ns, explicit, beta2, delta2, use48)
    np = length(om2); r32 = zeros(Int, np); r48 = zeros(Int, np)
    delta2 > 0 || return r32, r48
    total(mu) = sum(_best_split(om2[t], Ns[t], explicit[t], beta2[t], mu, use48)[2] for t in 1:np; init=0.0)
    mu = 0.0
    if total(0.0) > delta2
        mu = 1.0
        while total(mu) > delta2
            mu *= 1e3
        end
        mulo = mu
        while total(mulo) <= delta2
            mulo /= 1e3
        end
        for _ in 1:200
            mid = sqrt(mulo * mu)
            total(mid) > delta2 ? (mulo = mid) : (mu = mid)
            mu <= mulo * (1 + 1e-12) && break
        end
    end
    for t in 1:np
        r32[t], r48[t] = _best_split(om2[t], Ns[t], explicit[t], beta2[t], mu, use48)[1]
    end
    r32, r48
end

# Dense near-field blocks regrouped over the elementary row intervals induced
# by all block row ranges and leaf clusters (the leaves themselves for
# cluster-aligned blocks), so near packets write disjoint output rows. Blocks
# are split exactly into row slices.
function _leaf_near_groups(p::H2CompactMatvecPlan)
    cuts = Int[]
    for n in p.rows
        isempty(n.children) && !isempty(n.indices) && push!(cuts, first(n.indices), last(n.indices) + 1)
    end
    for b in p.dense
        isempty(b.rows) || push!(cuts, first(b.rows), last(b.rows) + 1)
    end
    sort!(unique!(cuts))
    groups = Dict{UnitRange{Int},Vector{Tuple{Int,UnitRange{Int}}}}(); order = UnitRange{Int}[]
    for (i, b) in enumerate(p.dense)
        R = b.rows; (isempty(R) || isempty(b.cols)) && continue
        c = searchsortedfirst(cuts, first(R))
        while c < length(cuts) && cuts[c] <= last(R)
            L = cuts[c]:cuts[c+1]-1
            haskey(groups, L) || (groups[L] = Tuple{Int,UnitRange{Int}}[]; push!(order, L))
            push!(groups[L], (i, L .- (first(R) - 1)))
            c += 1
        end
    end
    groups, order
end

function H2MixedPacketMatvecPlan(p::H2CompactMatvecPlan; workers::Int=1, precision_rtol::Real=1e-13, format48::Bool=false)
    workers > 0 || throw(ArgumentError("workers must be positive"))
    isfinite(precision_rtol) && precision_rtol >= 0 || throw(ArgumentError("precision_rtol must be finite and nonnegative"))
    groups = Dict{Int,Vector{Int}}(); order = Int[]
    for (i, b) in enumerate(p.couplings)
        isempty(p.rows[b.row].coeff) || isempty(p.cols[b.col].coeff) || begin
            haskey(groups, b.row) || (groups[b.row] = Int[]; push!(order, b.row))
            push!(groups[b.row], i)
        end
    end
    neargroups, nearorder = _leaf_near_groups(p)
    nf = length(order); nn = length(nearorder); np = nf + nn
    _coupling_matrix(b) = b isa _PlanCoupling ? b.S : (b.R === nothing ? b.L : b.L * b.R')
    # Packets 1:nf are far-field (one per row node), nf+1:np near-field (one per
    # leaf row range). Packets are processed block by block and never materialized.
    packet_blocks(t) = t <= nf ? [_coupling_matrix(p.couplings[i]) for i in groups[order[t]]] :
        [view(p.dense[i].D, rr, :) for (i, rr) in neargroups[nearorder[t-nf]]]
    function gram(blocks, k)
        G = zeros(k, k)
        for B in blocks
            mul!(G, B, B', 1.0, 1.0)
        end
        Symmetric(G)
    end
    packet_columns(t) = t <= nf ? [p.cols[p.couplings[i].col].coeff for i in groups[order[t]]] :
        [p.dense[i].cols for (i, _) in neargroups[nearorder[t-nf]]]
    rownorm = _compact_basis_norms(p.rows); colnorm = _compact_basis_norms(p.cols)
    columns = [packet_columns(t) for t in 1:np]
    Ns = [sum(length, columns[t]) for t in 1:np]
    outrows = [t <= nf ? p.rows[order[t]].coeff : nearorder[t-nf] for t in 1:np]
    ks = length.(outrows)
    beta2 = [t <= nf ? (rownorm[order[t]] * maximum(colnorm[p.couplings[i].col] for i in groups[order[t]]))^2 : 1.0 for t in 1:np]
    # Rotations of explicit far-field row bases are absorbed; implicit and physical rows store U.
    explicitU = [t > nf || p.rows[order[t]].identity for t in 1:np]
    # Pass 1: squared row weights ω² (eigenvalues of the Gram matrix Σ B Bᵀ over
    # the packet blocks, padded by an eigensolver error estimate). Only the
    # selection uses these weights; the reported bound uses the stored rows.
    om2 = [Float64[] for _ in 1:np]; fro2 = zeros(np)
    rot = precision_rtol > 0
    Threads.@threads :dynamic for t in 1:np
        blocks = packet_blocks(t); fro2[t] = sum(B -> sum(abs2, B), blocks)
        if rot
            λ = eigvals!(gram(blocks, ks[t]); alg=LinearAlgebra.DivideAndConquer())
            om2[t] = max.(λ[end:-1:1], 0.0) .+ 8 * eps() * max(λ[end], 0.0)
        end
    end
    eta = sqrt(sum(fro2; init=0.0))
    r32, r48 = rot ? _precision_selection(om2, Ns, explicitU, beta2, (precision_rtol * eta)^2, format48) : (zeros(Int, np), zeros(Int, np))
    n64 = ks .- r32 .- r48
    # A packet stored in a single format needs no rotation.
    rotated = [count(>(0), (n64[t], r48[t], r32[t])) > 1 for t in 1:np]
    # Pass 2: rotated packets split into Float64, optional 48-bit and Float32 rows.
    packets = Vector{_MixedPacket}(undef, np)
    lo2 = zeros(np); err2 = zeros(np); Uabs = Vector{Matrix{Float64}}(undef, np)
    Threads.@threads :dynamic for t in 1:np
        blocks = packet_blocks(t); k = ks[t]; N = Ns[t]; r1 = n64[t]; rm = r1 + r48[t]
        nref = r32[t] > 0 ? rm : r1
        hv = Float64[]; tau = Float64[]; U = zeros(0, 0); A = view(U, :, 1:0)
        if rotated[t]
            U = reverse!(eigen!(gram(blocks, k); alg=LinearAlgebra.DivideAndConquer()).vectors; dims=2)
            if !explicitU[t]
                Uabs[t] = U
            else
                # Householder QR of the leading singular vectors (in place): the
                # spans of Q[:, 1:j] and U[:, 1:j] agree for every j <= nref.
                A, tau = LAPACK.geqrf!(view(U, :, 1:nref))
                hv = reduce(vcat, [A[i+1:k, i] for i in 1:nref]; init=Float64[])
            end
        end
        hirows = _InterleavedRows{Float64}(undef, r1, N)
        midrows = _InterleavedRows{_T48}(undef, rm - r1, N); lorows = _InterleavedRows{Float32}(undef, k - rm, N)
        off = 0; c = 0.0; e = 0.0
        wbuf = rotated[t] ? Matrix{Float64}(undef, k, maximum(B -> size(B, 2), blocks)) : zeros(0, 0)
        for B in blocks
            Wb = if !rotated[t]
                B
            elseif !explicitU[t]
                mul!(view(wbuf, :, 1:size(B, 2)), U', B)
            else
                LAPACK.ormqr!('L', 'T', A, tau, copyto!(view(wbuf, :, 1:size(B, 2)), B))   # Qᵀ B, Q never formed
            end
            hiW = view(Wb, 1:r1, :); midW = view(Wb, r1+1:rm, :); loW = view(Wb, rm+1:k, :)
            _put_block!(hirows, hiW, off); _put_block!(midrows, midW, off); _put_block!(lorows, loW, off)
            c += _U32^2 * sum(abs2, loW) + length(loW) * 2.0^-300 + _U48^2 * sum(abs2, midW)
            e += _rounding_err2(lorows, loW, off) + _rounding_err2(midrows, midW, off)
            off += size(B, 2)
        end
        lo2[t] = beta2[t] * c; err2[t] = beta2[t] * e
        packets[t] = _MixedPacket(outrows[t], columns[t], hirows, midrows, lorows,
            hv, tau, zeros(N), zeros(isempty(tau) ? 0 : k))
    end
    # Absorb rotations of explicit (unsaturated) row bases into transfers and leaf bases.
    absorbed = Dict{Int,Matrix{Float64}}(order[t] => Uabs[t] for t in 1:nf if rotated[t] && !explicitU[t])
    rows = copy(p.rows)
    if !isempty(absorbed)
        parent = zeros(Int, length(rows))
        for (i, n) in enumerate(rows), j in n.children
            parent[j] = i
        end
        for (i, n) in enumerate(p.rows)
            own = get(absorbed, i, nothing)
            par = parent[i] == 0 ? nothing : get(absorbed, parent[i], nothing)
            (own === nothing && par === nothing) && continue
            E = n.E
            if !isempty(E)
                own === nothing || (E = own' * E)
                par === nothing || (E = E * par)
            end
            V = n.V
            if own !== nothing && !isempty(V)
                V = V * own
            end
            rows[i] = _CompactBasisNode(V, E, n.coeff, n.indices, n.children, n.identity)
        end
    end
    # Balanced static schedule over far and near packets (time model: Float32
    # elements stream about 1.4x faster than Float64 elements).
    cost = [b.hi.rows * b.hi.N + 1.35 * b.mid.rows * b.mid.N + 0.7 * b.lo.rows * b.lo.N + 4.0 * length(b.hv) + 2.0 * b.hi.N + 256.0 for b in packets]
    owner = _lpt_schedule(cost, workers)
    farschedule = [[t for t in 1:nf if owner[t] == w] for w in 1:workers]
    nearschedule = [[t - nf for t in nf+1:np if owner[t] == w] for w in 1:workers]
    sel(f, range) = sum((f(packets[t]) for t in range); init=0)
    precision = (; rtol=Float64(precision_rtol), reference_norm=eta,
        bound=sqrt(sum(lo2; init=0.0)), storage_perturbation=sqrt(sum(err2; init=0.0)),
        float32_rows=sum(r32; init=0), float48_rows=sum(r48; init=0), rows=sum(ks; init=0), rotated_packets=count(rotated), packets=np,
        near_rotated_packets=count(view(rotated, nf+1:np)), near_float32_rows=sum(view(r32, nf+1:np); init=0), absorbed_rotations=length(absorbed),
        float64_far_bytes=sum((8 * ks[t] * Ns[t] for t in 1:nf); init=0),
        float64_near_bytes=sum((8 * ks[t] * Ns[t] for t in nf+1:np); init=0),
        far_bytes=sel(_packet_bytes, 1:nf), near_bytes=sel(_packet_bytes, nf+1:np),
        float32_bytes=sel(b -> _nbytes(b.lo.data), 1:np), float48_bytes=sel(b -> _nbytes(b.mid.data), 1:np), rotation_bytes=sel(b -> sizeof(b.hv) + sizeof(b.tau), 1:np))
    H2MixedPacketMatvecPlan(p.shape, rows, p.cols, packets[1:nf], packets[nf+1:np], farschedule, nearschedule,
        zeros(length(p.rowcoeff)), zeros(length(p.colcoeff)), zeros(length(p.rowbuffer)), zeros(length(p.colbuffer)),
        p.rowperm, p.colperm, [zeros(length(p.colcoeff)) for _ in 1:workers], [zeros(length(p.colbuffer)) for _ in 1:workers],
        precision)
end
Base.size(p::H2MixedPacketMatvecPlan) = p.shape

function _mixed_forward!(rowcoeff::Vector{Float64}, b::_MixedPacket, colcoeff::Vector{Float64})
    s = b.scratch; off = 0
    @inbounds for cr in b.columns
        for (i, j) in enumerate(cr)
            s[off+i] = colcoeff[j]
        end
        off += length(cr)
    end
    if isempty(b.tau)
        zo = first(b.row) - 1
        _il_forward!(rowcoeff, zo, b.hi, s)
        _il_forward!(rowcoeff, zo + b.hi.rows, b.mid, s)
        _il_forward!(rowcoeff, zo + b.hi.rows + b.mid.rows, b.lo, s)
    else
        z = b.zbuf; fill!(z, 0.0)
        _il_forward!(z, 0, b.hi, s)
        _il_forward!(z, b.hi.rows, b.mid, s)
        _il_forward!(z, b.hi.rows + b.mid.rows, b.lo, s)
        _apply_reflectors!(z, b.hv, b.tau, false)
        zo = first(b.row) - 1
        @inbounds for i in eachindex(z)
            rowcoeff[zo+i] += z[i]
        end
    end
    nothing
end
function _mixed_transpose!(colcoeff::Vector{Float64}, b::_MixedPacket, rowcoeff::Vector{Float64})
    s = b.scratch; fill!(s, 0.0)
    if isempty(b.tau)
        wo = first(b.row) - 1
        _il_adjoint!(s, b.hi, rowcoeff, wo)
        _il_adjoint!(s, b.mid, rowcoeff, wo + b.hi.rows)
        _il_adjoint!(s, b.lo, rowcoeff, wo + b.hi.rows + b.mid.rows)
    else
        z = b.zbuf; zo = first(b.row) - 1
        @inbounds for i in eachindex(z)
            z[i] = rowcoeff[zo+i]
        end
        _apply_reflectors!(z, b.hv, b.tau, true)
        _il_adjoint!(s, b.hi, z, 0)
        _il_adjoint!(s, b.mid, z, b.hi.rows)
        _il_adjoint!(s, b.lo, z, b.hi.rows + b.mid.rows)
    end
    off = 0
    @inbounds for cr in b.columns
        for (i, j) in enumerate(cr)
            colcoeff[j] += s[off+i]
        end
        off += length(cr)
    end
    nothing
end
function _mixed_worker!(p::H2MixedPacketMatvecPlan, w::Int, t::Bool)
    single = length(p.partials) == 1
    farout = t ? (single ? p.colcoeff : p.partials[w]) : p.rowcoeff
    nearout = t ? (single ? p.colbuffer : p.nearpartials[w]) : p.rowbuffer
    for i in p.farschedule[w]
        b = p.packets[i]
        t ? _mixed_transpose!(farout, b, p.rowcoeff) : _mixed_forward!(farout, b, p.colcoeff)
    end
    for i in p.nearschedule[w]
        b = p.nearpackets[i]
        t ? _mixed_transpose!(nearout, b, p.rowbuffer) : _mixed_forward!(nearout, b, p.colbuffer)
    end
    nothing
end
# Far packets write row coefficients and near packets physical output rows, so
# both kinds run in one parallel phase; transposes use private partial sums.
function _mixed_interactions!(p::H2MixedPacketMatvecPlan, t::Bool)
    if length(p.partials) == 1
        _mixed_worker!(p, 1, t)
        return nothing
    end
    if t
        foreach(v -> fill!(v, 0.0), p.partials); foreach(v -> fill!(v, 0.0), p.nearpartials)
    end
    @sync for w in eachindex(p.partials)
        Threads.@spawn _mixed_worker!(p, w, t)
    end
    if t
        for partial in p.partials
            for i in eachindex(p.colcoeff)
                p.colcoeff[i] += partial[i]
            end
        end
        for partial in p.nearpartials
            for i in eachindex(p.colbuffer)
                p.colbuffer[i] += partial[i]
            end
        end
    end
    nothing
end
function _mixed_mul!(y, p::H2MixedPacketMatvecPlan, x, alpha, beta, t)
    inputnodes, outputnodes = t ? (p.rows, p.cols) : (p.cols, p.rows)
    inputcoeff, outputcoeff = t ? (p.rowcoeff, p.colcoeff) : (p.colcoeff, p.rowcoeff)
    input, output = t ? (p.rowbuffer, p.colbuffer) : (p.colbuffer, p.rowbuffer)
    ip, op = t ? (p.rowperm, p.colperm) : (p.colperm, p.rowperm)
    length(x) == length(input) && length(y) == length(output) || throw(DimensionMismatch("incompatible mixed-precision H2 matvec dimensions"))
    if iszero(alpha)
        iszero(beta) ? fill!(y, 0.0) : rmul!(y, beta)
        return y
    end
    for i in eachindex(input)
        input[i] = x[ip[i]]
    end
    fill!(output, 0.0); fill!(outputcoeff, 0.0)
    _plan_up!(inputcoeff, inputnodes, input)
    _mixed_interactions!(p, t)
    _plan_down!(output, outputcoeff, outputnodes)
    for i in eachindex(output)
        j = op[i]; y[j] = iszero(beta) ? alpha * output[i] : alpha * output[i] + beta * y[j]
    end
    y
end
const TransposedMixedH2Plan = Union{Transpose{Float64,H2MixedPacketMatvecPlan},Adjoint{Float64,H2MixedPacketMatvecPlan}}
LinearAlgebra.mul!(y::AbstractVector, p::H2MixedPacketMatvecPlan, x::AbstractVector, alpha::Number=1, beta::Number=0) = _mixed_mul!(y, p, x, alpha, beta, false)
LinearAlgebra.mul!(y::AbstractVector, p::TransposedMixedH2Plan, x::AbstractVector, alpha::Number=1, beta::Number=0) = _mixed_mul!(y, parent(p), x, alpha, beta, true)
Base.:*(p::Union{H2MixedPacketMatvecPlan,TransposedMixedH2Plan}, x::AbstractVector) = mul!(zeros(size(p, 1)), p, x)
function storage_bytes(p::H2MixedPacketMatvecPlan)
    sum((sizeof(n.V) + sizeof(n.E) for ns in (p.rows, p.cols) for n in ns); init=0) +
    sum((_packet_bytes(b) for b in p.nearpackets); init=0) + sum((_packet_bytes(b) for b in p.packets); init=0)
end
function Base.copy(p::H2MixedPacketMatvecPlan)
    H2MixedPacketMatvecPlan(p.shape, p.rows, p.cols, map(_copy_packet, p.packets), map(_copy_packet, p.nearpackets), p.farschedule, p.nearschedule,
        zeros(length(p.rowcoeff)), zeros(length(p.colcoeff)), zeros(length(p.rowbuffer)), zeros(length(p.colbuffer)), p.rowperm, p.colperm,
        [zeros(length(v)) for v in p.partials], [zeros(length(v)) for v in p.nearpartials], p.precision)
end
"""
    precision_summary(plan::H2MixedPacketMatvecPlan)

Selection and error bounds of a mixed-precision plan: `rtol` and
`reference_norm` (η), the a priori Frobenius bound `bound` (≤ rtol·η), the
exactly computed Frobenius norm of the Float32 rounding of the stored rows
(`storage_perturbation`, scaled by basis norms), the numbers of Float32 rows
and rotated packets, and byte counts.
"""
precision_summary(p::H2MixedPacketMatvecPlan) = p.precision
Base.show(io::IO, p::H2MixedPacketMatvecPlan) = print(io, "H2MixedPacketMatvecPlan(", size(p, 1), " × ", size(p, 2), ", ",
    length(p.partials), " workers, ", storage_bytes(p), " numeric bytes, precision_rtol=", p.precision.rtol, ")")
Base.show(io::IO, ::MIME"text/plain", p::H2MixedPacketMatvecPlan) = show(io, p)
