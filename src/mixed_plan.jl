# Adaptive mixed-precision packet plan.
#
# Each coupling packet P_t (all couplings of one row node, as in
# H2PacketMatvecPlan) is rotated to the left singular basis of P_t,
# P_t = U_t W_t, so that row i of W_t has norm omega_i ~ sigma_i(P_t).
# Rows with small omega_i are stored in Float32 and all products accumulate in
# Float64. Rounding row i to Float32 perturbs the operator by at most
# u32*omega_i in the Frobenius norm, so precision follows singular-value
# weight; the leading rows stay Float64.
#
# Rows of one packet are stored in an interleaved layout (groups of four rows,
# column-interleaved), so each packet is one sequential memory stream. The
# kernels read it with SIMD and accumulate in Float64. The packets run on the
# task engine of H2PacketMatvecPlan (phases, write ownership, slot reductions,
# near-field column packets), so products are deterministic and bitwise
# independent of the worker count.

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
# Largest packet Frobenius norm whose rows may be stored in each reduced format:
# every stored entry is bounded by the norm (up to rotation roundoff), so the
# factors keep rounding (and the 48-bit carry) away from overflow.
const _MAX32 = Float64(floatmax(Float32)) / 2
const _MAX48 = floatmax(Float64) / 4

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
@inline function _il_fwd4!(z, zo, d, o, N, x, xo)
    s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0
    @inbounds @simd for j in 1:N
        xj = x[xo+j]; q = o + 4(j - 1)
        s1 = muladd(_ld(d, q+1), xj, s1); s2 = muladd(_ld(d, q+2), xj, s2)
        s3 = muladd(_ld(d, q+3), xj, s3); s4 = muladd(_ld(d, q+4), xj, s4)
    end
    @inbounds begin
        z[zo+1] += s1; z[zo+2] += s2; z[zo+3] += s3; z[zo+4] += s4
    end
    nothing
end
@inline function _il_fwd1!(z, zo, d, o, N, x, xo)
    s = 0.0
    @inbounds @simd for j in 1:N
        s = muladd(_ld(d, o+j), x[xo+j], s)
    end
    @inbounds z[zo+1] += s
    nothing
end
# z[zo+i] += sum_j A[i,j] * x[xo+j]
function _il_forward!(z::Vector{Float64}, zo::Int, A::_InterleavedRows, x::Vector{Float64}, xo::Int=0)
    N = A.N; g4 = A.rows ÷ 4; d = A.data
    for g in 0:g4-1
        _il_fwd4!(z, zo + 4g, d, 4N * g, N, x, xo)
    end
    for r in 1:A.rows%4
        _il_fwd1!(z, zo + 4g4 + r - 1, d, 4N * g4 + N * (r - 1), N, x, xo)
    end
    nothing
end
@inline function _il_adj4!(x, xo, d, o, N, w, wo)
    @inbounds w1 = w[wo+1]; @inbounds w2 = w[wo+2]; @inbounds w3 = w[wo+3]; @inbounds w4 = w[wo+4]
    @inbounds @simd for j in 1:N
        q = o + 4(j - 1)
        x[xo+j] = muladd(_ld(d, q+4), w4, muladd(_ld(d, q+3), w3, muladd(_ld(d, q+2), w2, muladd(_ld(d, q+1), w1, x[xo+j]))))
    end
    nothing
end
@inline function _il_adj1!(x, xo, d, o, N, wi)
    @inbounds @simd for j in 1:N
        x[xo+j] = muladd(_ld(d, o+j), wi, x[xo+j])
    end
    nothing
end
# x[xo+j] += sum_i A[i,j] * w[wo+i]   (exact transpose of _il_forward! on the same stored data)
function _il_adjoint!(x::Vector{Float64}, xo::Int, A::_InterleavedRows, w::Vector{Float64}, wo::Int)
    N = A.N; g4 = A.rows ÷ 4; d = A.data
    for g in 0:g4-1
        _il_adj4!(x, xo, d, 4N * g, N, w, wo + 4g)
    end
    for r in 1:A.rows%4
        @inbounds wi = w[wo+4g4+r]
        _il_adj1!(x, xo, d, 4N * g4 + N * (r - 1), N, wi)
    end
    nothing
end

# Several right-hand sides: register blocks of R stored rows (4 or 1) × V vectors.
# Z[zo+r+(v-1)lz] += Σ_j A[r,j] X[xo+j+(v-1)lx]
@generated function _il_fwdk!(Z, zo, lz, d, o, N, X, xo, lx, ::Val{R}, ::Val{V}) where {R,V}
    acc(r, v) = Symbol(:s_, r, :_, v)
    init = [:($(acc(r, v)) = 0.0) for r in 1:R for v in 1:V]
    loads = [:($(Symbol(:a_, r)) = _ld(d, q + $r)) for r in 1:R]
    xs = [:($(Symbol(:x_, v)) = X[xo+j+$(v - 1)*lx]) for v in 1:V]
    fm = [:($(acc(r, v)) = muladd($(Symbol(:a_, r)), $(Symbol(:x_, v)), $(acc(r, v)))) for r in 1:R for v in 1:V]
    st = [:(Z[zo+$r+$(v - 1)*lz] += $(acc(r, v))) for r in 1:R for v in 1:V]
    quote
        $(init...)
        @inbounds @simd for j in 1:N
            q = o + $R * (j - 1)
            $(loads...)
            $(xs...)
            $(fm...)
        end
        @inbounds begin
            $(st...)
        end
        nothing
    end
end
# X[xo+j+(v-1)lx] += Σ_r A[r,j] W[wo+r+(v-1)lw]
@generated function _il_adjk!(X, xo, lx, d, o, N, W, wo, lw, ::Val{R}, ::Val{V}) where {R,V}
    wv(r, v) = Symbol(:w_, r, :_, v)
    ws = [:($(wv(r, v)) = W[wo+$r+$(v - 1)*lw]) for r in 1:R for v in 1:V]
    loads = [:($(Symbol(:a_, r)) = _ld(d, q + $r)) for r in 1:R]
    body = Expr[]
    for v in 1:V
        idx = :(xo + j + $(v - 1) * lx); ex = :(X[$idx])
        for r in 1:R
            ex = :(muladd($(Symbol(:a_, r)), $(wv(r, v)), $ex))
        end
        push!(body, :(X[$idx] = $ex))
    end
    quote
        @inbounds begin
            $(ws...)
        end
        @inbounds @simd for j in 1:N
            q = o + $R * (j - 1)
            $(loads...)
            $(body...)
        end
        nothing
    end
end
@inline function _il_fwdk_call!(Z, zo, lz, d, o, N, X, xo, lx, r::Val, v)
    v == 4 ? _il_fwdk!(Z, zo, lz, d, o, N, X, xo, lx, r, Val(4)) : v == 3 ? _il_fwdk!(Z, zo, lz, d, o, N, X, xo, lx, r, Val(3)) :
    v == 2 ? _il_fwdk!(Z, zo, lz, d, o, N, X, xo, lx, r, Val(2)) : _il_fwdk!(Z, zo, lz, d, o, N, X, xo, lx, r, Val(1))
end
@inline function _il_adjk_call!(X, xo, lx, d, o, N, W, wo, lw, r::Val, v)
    v == 4 ? _il_adjk!(X, xo, lx, d, o, N, W, wo, lw, r, Val(4)) : v == 3 ? _il_adjk!(X, xo, lx, d, o, N, W, wo, lw, r, Val(3)) :
    v == 2 ? _il_adjk!(X, xo, lx, d, o, N, W, wo, lw, r, Val(2)) : _il_adjk!(X, xo, lx, d, o, N, W, wo, lw, r, Val(1))
end
# Z[zo+i+(v-1)lz] += Σ_j A[i,j] X[xo+j+(v-1)lx] for v=1:K. Each four-row group
# is streamed once per register block of vectors, while it is still cached.
function _il_forward_k!(Z::Vector{Float64}, zo::Int, lz::Int, A::_InterleavedRows, X::Vector{Float64}, xo::Int, lx::Int, K::Int)
    N = A.N; g4 = A.rows ÷ 4; d = A.data
    for g in 0:g4-1
        v = 0
        while v < K
            c = _vblock(K - v); _il_fwdk_call!(Z, zo + 4g + v * lz, lz, d, 4N * g, N, X, xo + v * lx, lx, Val(4), c); v += c
        end
    end
    for r in 1:A.rows%4
        v = 0
        while v < K
            c = _vblock(K - v); _il_fwdk_call!(Z, zo + 4g4 + r - 1 + v * lz, lz, d, 4N * g4 + N * (r - 1), N, X, xo + v * lx, lx, Val(1), c); v += c
        end
    end
    nothing
end
# X[xo+j+(v-1)lx] += Σ_i A[i,j] W[wo+i+(v-1)lw] for v=1:K.
function _il_adjoint_k!(X::Vector{Float64}, xo::Int, lx::Int, A::_InterleavedRows, W::Vector{Float64}, wo::Int, lw::Int, K::Int)
    N = A.N; g4 = A.rows ÷ 4; d = A.data
    for g in 0:g4-1
        v = 0
        while v < K
            c = _vblock(K - v); _il_adjk_call!(X, xo + v * lx, lx, d, 4N * g, N, W, wo + 4g + v * lw, lw, Val(4), c); v += c
        end
    end
    for r in 1:A.rows%4
        v = 0
        while v < K
            c = _vblock(K - v); _il_adjk_call!(X, xo + v * lx, lx, d, 4N * g4 + N * (r - 1), N, W, wo + 4g4 + r - 1 + v * lw, lw, Val(1), c); v += c
        end
    end
    nothing
end

# Coupling packet of the mixed plan: rows 1:hi.rows in Float64, then 48-bit,
# then Float32 rows, all over the packet's N columns (colcoeff segments
# `columns`, adjoint slots `slot+1:slot+N`).
struct _MixedCouplingPacket
    row::UnitRange{Int}
    columns::Vector{UnitRange{Int}}
    slot::Int
    hi::_Rows64
    mid::_Rows48
    lo::_Rows32
    # Explicit packet rotation Q = H_1 ⋯ H_r as Householder reflectors (empty if
    # none or absorbed into the row basis): reflector i is [1; hv[off_i+1:off_i+k-i]].
    hv::Vector{Float64}
    tau::Vector{Float64}
end
_packet_ncols(b::_MixedCouplingPacket) = b.hi.N
_packet_bytes(b::_MixedCouplingPacket) = _nbytes(b.hi.data) + _nbytes(b.mid.data) + _nbytes(b.lo.data) + sizeof(b.hv) + sizeof(b.tau)
# Time model for task ordering: Float32 elements stream about 1.4x faster than Float64 ones.
_packet_work(b::_MixedCouplingPacket) =
    b.hi.rows * b.hi.N + 1.35 * b.mid.rows * b.mid.N + 0.7 * b.lo.rows * b.lo.N + 4.0 * length(b.hv) + 2.0 * b.hi.N
# Per-worker scratch (per right-hand side): gathered input and rotated rows.
_packet_scratch(b::_MixedCouplingPacket) = b.hi.N + (isempty(b.tau) ? 0 : length(b.row))
# Stored Float64 entries of r Householder reflectors of a k-row rotation.
_reflector_entries(k, r) = r * k - (r * (r + 1)) ÷ 2
@inline function _reflect!(z, zo, hv, off, i, k, τ)
    m = k - i; s = @inbounds z[zo+i]
    @inbounds @simd for l in 1:m
        s = muladd(hv[off+l], z[zo+i+l], s)
    end
    s *= τ
    @inbounds z[zo+i] -= s
    @inbounds @simd for l in 1:m
        z[zo+i+l] = muladd(-s, hv[off+l], z[zo+i+l])
    end
    nothing
end
# z[zo+1:zo+k] <- Q z (transposed=false) or Q' z, Q = H_1 ⋯ H_r; each H_i is
# symmetric, so both directions apply exactly the same stored reflectors.
function _apply_reflectors!(z::Vector{Float64}, zo::Int, k::Int, hv::Vector{Float64}, tau::Vector{Float64}, transposed::Bool)
    r = length(tau)
    if transposed
        off = 0
        for i in 1:r
            _reflect!(z, zo, hv, off, i, k, tau[i]); off += k - i
        end
    else
        for i in r:-1:1
            _reflect!(z, zo, hv, (i - 1) * k - ((i - 1) * i) ÷ 2, i, k, tau[i])
        end
    end
    nothing
end
_apply_reflectors!(z::Vector{Float64}, hv::Vector{Float64}, tau::Vector{Float64}, transposed::Bool) =
    _apply_reflectors!(z, 0, length(z), hv, tau, transposed)

# Forward: rowcoeff[row] += Q W colcoeff[columns]; adjoint: slot = Wᵀ Qᵀ rowcoeff[row].
function _coupling_task!(p, b::_MixedCouplingPacket, w, t)
    s = p.scratch[w]; N = b.hi.N; k = length(b.row); r0 = first(b.row) - 1
    o2 = b.hi.rows; o3 = o2 + b.mid.rows
    if t
        sl = p.slots; s0 = b.slot
        @inbounds for j in 1:N; sl[s0+j] = 0.0; end
        if isempty(b.tau)
            src = p.rowcoeff; wo = r0
        else
            @inbounds for i in 1:k; s[i] = p.rowcoeff[r0+i]; end
            _apply_reflectors!(s, 0, k, b.hv, b.tau, true)
            src = s; wo = 0
        end
        _il_adjoint!(sl, s0, b.hi, src, wo)
        _il_adjoint!(sl, s0, b.mid, src, wo + o2)
        _il_adjoint!(sl, s0, b.lo, src, wo + o3)
    else
        off = 0; x = p.colcoeff
        @inbounds for cr in b.columns
            c0 = first(cr) - 1
            for i in 1:length(cr); s[off+i] = x[c0+i]; end
            off += length(cr)
        end
        if isempty(b.tau)
            z = p.rowcoeff; zo = r0
        else
            z = s; zo = N
            @inbounds for i in 1:k; s[N+i] = 0.0; end
        end
        _il_forward!(z, zo, b.hi, s, 0)
        _il_forward!(z, zo + o2, b.mid, s, 0)
        _il_forward!(z, zo + o3, b.lo, s, 0)
        if !isempty(b.tau)
            _apply_reflectors!(s, N, k, b.hv, b.tau, false)
            y = p.rowcoeff
            @inbounds for i in 1:k; y[r0+i] += s[N+i]; end
        end
    end
    nothing
end
function _coupling_task_k!(p, ws, b::_MixedCouplingPacket, w, t, K)
    lrc = length(p.rowcoeff); lcc = length(p.colcoeff); ls = length(p.slots)
    s = ws.scratch[w]; N = b.hi.N; k = length(b.row); r0 = first(b.row) - 1
    o2 = b.hi.rows; o3 = o2 + b.mid.rows
    if t
        sl = ws.slots; s0 = b.slot
        @inbounds for v in 0:K-1, j in 1:N; sl[s0+j+v*ls] = 0.0; end
        if isempty(b.tau)
            src = ws.rowcoeff; wo = r0; lw = lrc
        else
            for v in 0:K-1
                @inbounds for i in 1:k; s[i+v*k] = ws.rowcoeff[r0+i+v*lrc]; end
                _apply_reflectors!(s, v * k, k, b.hv, b.tau, true)
            end
            src = s; wo = 0; lw = k
        end
        _il_adjoint_k!(sl, s0, ls, b.hi, src, wo, lw, K)
        _il_adjoint_k!(sl, s0, ls, b.mid, src, wo + o2, lw, K)
        _il_adjoint_k!(sl, s0, ls, b.lo, src, wo + o3, lw, K)
    else
        x = ws.colcoeff
        for v in 0:K-1
            off = v * N
            @inbounds for cr in b.columns
                c0 = first(cr) - 1 + v * lcc
                for i in 1:length(cr); s[off+i] = x[c0+i]; end
                off += length(cr)
            end
        end
        if isempty(b.tau)
            z = ws.rowcoeff; zo = r0; lz = lrc
        else
            z = s; zo = N * K; lz = k
            @inbounds for i in 1:k*K; s[N*K+i] = 0.0; end
        end
        _il_forward_k!(z, zo, lz, b.hi, s, 0, N, K)
        _il_forward_k!(z, zo + o2, lz, b.mid, s, 0, N, K)
        _il_forward_k!(z, zo + o3, lz, b.lo, s, 0, N, K)
        if !isempty(b.tau)
            y = ws.rowcoeff
            for v in 0:K-1
                _apply_reflectors!(s, N * K + v * k, k, b.hv, b.tau, false)
                @inbounds for i in 1:k; y[r0+i+v*lrc] += s[N*K+v*k+i]; end
            end
        end
    end
    nothing
end

"""
    H2MixedPacketMatvecPlan(compact_plan; workers=1, precision_rtol=1e-13, format48=false)
    H2MixedPacketMatvecPlan(h2; workers=1, precision_rtol=1e-13, format48=false, consume=false)

Packet plan with adaptive mixed-precision storage. Coupling packets hold all
couplings of one row node and the near field is stored in Float64 column
packets, exactly as in [`H2PacketMatvecPlan`](@ref), whose task engine runs
the products: the same phases and write ownership, so products are
deterministic and bitwise independent of the worker count, `mul!(Y, plan, X)`
with matrices applies several right-hand sides while streaming the stored
operator once, and `copy(plan)` shares the numerical data and gives private
scratch. Use BLAS threads=1 with `workers>1`.

Each coupling packet `P` (`k × N`) is rotated to its left singular basis,
`P = Q W`. Row `i` of `W` then has norm `ω_i = σ_i(P)`. Trailing rows with
small weight are stored in Float32 and the leading rows in Float64, row-
interleaved. Every product accumulates in Float64; Float32 values are only
widened, never used as accumulators. Rotations of explicit (unsaturated) row
bases are absorbed into copies of the transfers and leaf bases, so they cost
no storage. For implicit (saturated) or physical rows, `Q` is stored as the
`r` Householder reflectors spanning the rows above the lowest-precision block
(about `8r(k - r/2)` bytes). That cost enters the selection. The rounding cost
depends only on these subspaces, not on the basis chosen inside them.

With `format48=true`, rows of intermediate weight can also use a 48-bit format:
a Float64 rounded to 36 mantissa bits (unit roundoff `u₄₈ = 2⁻³⁷`), stored as
a UInt32 plane plus a UInt16 plane. This lowers storage further, but its
decode makes those rows slower to stream than Float64 rows on CPUs where a
few workers already saturate memory bandwidth.

A global Lagrangian allocation picks the rows. It minimizes stored bytes
subject to the a priori bound

    ‖Ã - A‖₂ ≤ ‖Ã - A‖_F ≤ (Σ_t β_t² [Σ_{i∈f32(t)} (u₃₂² ω_i² + N_t 2⁻³⁰⁰) + Σ_{i∈f48(t)} u₄₈² ω_i²])^{1/2} ≤ precision_rtol · η.

Here `A` is the Float64 compact operator and `Ã` the stored mixed operator
with the rotations applied in exact arithmetic, and `u₃₂ = 2⁻²⁴`. `β_t` is the
product of the row- and column-basis 2-norms (1 for orthonormal, implicit or
physical bases). `η = (Σ ‖stored block‖_F²)^{1/2}` over all coupling and
near-field blocks; it equals `‖A‖_F` only for orthonormal bases (otherwise
`precision_rtol` is relative to `η`, not to `‖A‖_F`). The bound is additive
over packets because packets cover disjoint matrix blocks; the `2⁻³⁰⁰` terms
cover Float32 underflow. It covers the reduced-precision rounding only: the
Float64 rotations themselves (`fl(QᵀP)`, `fl(UᵀE)`, `fl(VU)`) add roundoff of
order `ε₆₄ · κ · ‖P‖` that the bound does not include. For orthonormal bases
that is about `1e-16` relative and immaterial for `precision_rtol ≥ 1e-14`; with
ill-conditioned non-orthonormal bases the measured perturbation can exceed the
reported bound while staying at Float64 roundoff level relative to `‖A‖`.
For Gaussian inputs the bound limits the root-mean-square relative
perturbation of products: `E‖(Ã-A)x‖² / E‖Ax‖² ≤ precision_rtol²`; it is not a
per-vector guarantee. A packet is eligible for Float32 (48-bit) rows only if
its Frobenius norm is below `floatmax(Float32)/2` (`floatmax(Float64)/4`), so
reduced-precision rows cannot overflow; an operator whose `η` is not finite is
stored entirely in Float64. `precision_rtol=0` keeps everything in Float64 and
differs from `H2PacketMatvecPlan` only by the summation order of the
interleaved coupling kernels. Near-field blocks always stay in Float64 (their
spectra are flat, so rotating them saves almost nothing).

Forward and adjoint products apply the same stored values, so
`adjoint(plan)` is the exact transpose of the stored mixed operator.
Construction is bitwise deterministic for any number of Julia threads.
`consume=true` (H2 input only) releases the source operator's blocks while
the packets are built, as in `H2PacketMatvecPlan`; the source `h2` is
unusable afterwards. `precision_summary(plan)` reports the selection and both
bounds.
"""
struct H2MixedPacketMatvecPlan <: AbstractMatrix{Float64}
    engine::H2PacketMatvecPlan{_MixedCouplingPacket}
    precision::NamedTuple
end
function H2MixedPacketMatvecPlan(h::H2Matrix; workers::Int=1, precision_rtol::Real=1e-13, format48::Bool=false,
                                 consume::Bool=false)
    core = H2CompactMatvecPlan(h)
    # The compact plan references every numerical block it needs (see H2PacketMatvecPlan).
    consume && _release_h2_blocks!(h)
    H2MixedPacketMatvecPlan(core; workers, precision_rtol, format48, _release=consume)
end

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

# Rows sorted by decreasing weight are stored as Float64 rows 1:m₄₈, optional
# 48-bit rows m₄₈+1:m and Float32 rows m+1:k. For multiplier mu this returns the
# split (r32 = k-m, r48 = m-m₄₈) maximizing saved bytes - mu*cost and its cost.
# An explicit rotation stores the reflectors spanning the rows above the last
# format boundary; any orthonormal basis of the remaining complement gives the
# same rounding cost, because the cost depends only on the subspaces.
# `allow32 = false` excludes Float32 rows (overflow guard).
function _best_split(om2::Vector{Float64}, N::Int, explicit::Bool, beta2::Float64, mu::Float64, use48::Bool, allow32::Bool=true)
    k = length(om2)
    m48 = use48 ? count(w -> 2.0 * N <= mu * beta2 * _U48^2 * w, om2) : k   # rows that prefer Float64 over 48-bit
    p48 = zeros(k + 1)
    for i in 1:k
        p48[i+1] = p48[i] + beta2 * _U48^2 * om2[i]
    end
    best = 0.0; bestr = (0, 0); bestc = 0.0; c32 = 0.0
    for r32 in 0:(allow32 ? k : 0)
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
function _precision_selection(om2, Ns, explicit, beta2, delta2, use48, allow32=fill(true, length(om2)))
    np = length(om2); r32 = zeros(Int, np); r48 = zeros(Int, np)
    delta2 > 0 || return r32, r48
    split(t, mu) = _best_split(om2[t], Ns[t], explicit[t], beta2[t], mu, use48[t], allow32[t])
    total(mu) = sum(split(t, mu)[2] for t in 1:np; init=0.0)
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
        r32[t], r48[t] = split(t, mu)[1]
    end
    r32, r48
end

function H2MixedPacketMatvecPlan(p::H2CompactMatvecPlan; workers::Int=1, precision_rtol::Real=1e-13, format48::Bool=false,
                                 _release::Bool=false)
    workers > 0 || throw(ArgumentError("workers must be positive"))
    isfinite(precision_rtol) && precision_rtol >= 0 || throw(ArgumentError("precision_rtol must be finite and nonnegative"))
    tracker = _release ? _ReleaseTracker(storage_bytes(p)) : nothing
    groups, order = _coupling_groups(p, _release)
    nf = length(order)
    # Packets are processed block by block and never materialized.
    packet_blocks(t) = [_coupling_matrix(p.couplings[i]) for i in groups[order[t]]]
    gram(blocks, k) = (G = zeros(k, k); foreach(B -> mul!(G, B, B', 1.0, 1.0), blocks); Symmetric(G))
    columns = [[p.cols[p.couplings[i].col].coeff for i in groups[order[t]]] for t in 1:nf]
    Ns = [sum(length, columns[t]; init=0) for t in 1:nf]
    ks = [length(p.rows[order[t]].coeff) for t in 1:nf]
    rownorm = _compact_basis_norms(p.rows); colnorm = _compact_basis_norms(p.cols)
    beta2 = [(rownorm[order[t]] * maximum(colnorm[p.couplings[i].col] for i in groups[order[t]]))^2 for t in 1:nf]
    # Rotations of explicit row bases are absorbed; implicit (saturated) rows store U.
    explicitU = [p.rows[order[t]].identity for t in 1:nf]
    # Pass 1: squared row weights ω² (eigenvalues of the Gram matrix Σ B Bᵀ over
    # the packet blocks, padded by an eigensolver error estimate). Only the
    # selection uses these weights; the reported bound uses the stored rows.
    om2 = [Float64[] for _ in 1:nf]; fro2 = zeros(nf)
    Threads.@threads :dynamic for t in 1:nf
        blocks = packet_blocks(t); fro2[t] = sum(B -> sum(abs2, B), blocks; init=0.0)
        if precision_rtol > 0 && isfinite(fro2[t])
            λ = eigvals!(gram(blocks, ks[t]); alg=LinearAlgebra.DivideAndConquer())
            om2[t] = max.(λ[end:-1:1], 0.0) .+ 8 * eps() * max(λ[end], 0.0)
        end
    end
    nearfro2 = [isempty(b.rows) || isempty(b.cols) ? 0.0 : sum(abs2, b.D) for b in p.dense]
    eta = sqrt(sum(fro2; init=0.0) + sum(nearfro2; init=0.0))
    rot = precision_rtol > 0 && isfinite(eta)
    # Overflow guard: every stored entry of a rotated packet is bounded by its
    # Frobenius norm (up to rotation roundoff).
    allow32 = [sqrt(f) <= _MAX32 for f in fro2]
    use48 = [format48 && sqrt(f) <= _MAX48 for f in fro2]
    r32, r48 = rot ? _precision_selection(om2, Ns, explicitU, beta2, (precision_rtol * eta)^2, use48, allow32) : (zeros(Int, nf), zeros(Int, nf))
    n64 = ks .- r32 .- r48
    # A packet stored in a single format needs no rotation.
    rotated = [count(>(0), (n64[t], r48[t], r32[t])) > 1 for t in 1:nf]
    slots = cumsum(vcat(0, Ns))
    # Pass 2: rotated packets split into Float64, optional 48-bit and Float32 rows.
    packets = Vector{_MixedCouplingPacket}(undef, nf)
    lo2 = zeros(nf); err2 = zeros(nf)
    # Rotations of explicit (unsaturated) row bases are absorbed into copied
    # basis nodes as soon as their packet is built: node i gets E ← Uᵢᵀ E,
    # V ← V Uᵢ, and its children E ← E Uᵢ. Under a lock, and in a fixed order
    # whatever the thread schedule: a child's own left factor is always applied
    # before its parent's right factor (held until then), so E_c = (U_cᵀ E_c) U_p
    # bitwise for any number of threads.
    rows = copy(p.rows); absorbing = ReentrantLock(); nabsorbed = Threads.Atomic{Int}(0)
    willabsorb = falses(length(rows))
    for t in 1:nf
        rotated[t] && !explicitU[t] && (willabsorb[order[t]] = true)
    end
    leftdone = falses(length(rows)); heldright = Vector{Union{Nothing,Matrix{Float64}}}(nothing, length(rows))
    function absorb!(i, U)
        lock(absorbing) do
            n = rows[i]
            E = isempty(n.E) ? n.E : U' * n.E
            V = isempty(n.V) ? n.V : n.V * U
            R = heldright[i]
            if R !== nothing
                isempty(E) || (E = E * R)
                heldright[i] = nothing
            end
            rows[i] = _CompactBasisNode(V, E, n.coeff, n.indices, n.children, n.identity)
            leftdone[i] = true
            for j in n.children
                c = rows[j]
                isempty(c.E) && continue
                if willabsorb[j] && !leftdone[j]
                    heldright[j] = U
                else
                    rows[j] = _CompactBasisNode(c.V, c.E * U, c.coeff, c.indices, c.children, c.identity)
                end
            end
        end
        Threads.atomic_add!(nabsorbed, 1)
        nothing
    end
    Threads.@threads :dynamic for t in 1:nf
        blocks = packet_blocks(t); k = ks[t]; N = Ns[t]; r1 = n64[t]; rm = r1 + r48[t]
        nref = r32[t] > 0 ? rm : r1
        hv = Float64[]; tau = Float64[]; U = zeros(0, 0); A = view(U, :, 1:0)
        if rotated[t]
            U = reverse!(eigen!(gram(blocks, k); alg=LinearAlgebra.DivideAndConquer()).vectors; dims=2)
            if explicitU[t]
                # Householder QR of the leading singular vectors (in place): the
                # spans of Q[:, 1:j] and U[:, 1:j] agree for every j <= nref.
                A, tau = LAPACK.geqrf!(view(U, :, 1:nref))
                hv = reduce(vcat, [A[i+1:k, i] for i in 1:nref]; init=Float64[])
            end
        end
        hirows = _InterleavedRows{Float64}(undef, r1, N)
        midrows = _InterleavedRows{_T48}(undef, rm - r1, N); lorows = _InterleavedRows{Float32}(undef, k - rm, N)
        off = 0; c = 0.0; e = 0.0; released = 0
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
            off += size(B, 2); released += sizeof(B)
        end
        # The eligibility guard makes this unreachable; never store Inf silently.
        isfinite(e) || error("H2MixedPacketMatvecPlan: non-finite reduced-precision rounding in packet $t")
        lo2[t] = beta2[t] * c; err2[t] = beta2[t] * e
        packets[t] = _MixedCouplingPacket(p.rows[order[t]].coeff, columns[t], slots[t], hirows, midrows, lorows, hv, tau)
        if _release
            empty!(blocks)
            for i in groups[order[t]]; p.couplings[i] = _released_coupling(p.couplings[i]); end
            _released!(tracker, released)
        end
        rotated[t] && !explicitU[t] && absorb!(order[t], U)
    end
    all(isnothing, heldright) || error("H2MixedPacketMatvecPlan: unabsorbed basis rotation")
    engine = _packet_engine(p, rows, packets, workers, _release, tracker)
    sel(f) = sum((f(b) for b in packets); init=0)
    precision = (; rtol=Float64(precision_rtol), reference_norm=eta,
        bound=sqrt(sum(lo2; init=0.0)), storage_perturbation=sqrt(sum(err2; init=0.0)),
        float32_rows=sum(r32; init=0), float48_rows=sum(r48; init=0), rows=sum(ks; init=0), rotated_packets=count(rotated),
        packets=nf, near_packets=length(engine.nearpackets), absorbed_rotations=nabsorbed[],
        float32_ineligible_packets=count(!, allow32),
        float64_far_bytes=sum((8 * ks[t] * Ns[t] for t in 1:nf); init=0),
        far_bytes=sel(_packet_bytes), near_bytes=sum((sizeof(b.matrix) for b in engine.nearpackets); init=0),
        float32_bytes=sel(b -> _nbytes(b.lo.data)), float48_bytes=sel(b -> _nbytes(b.mid.data)),
        rotation_bytes=sel(b -> sizeof(b.hv) + sizeof(b.tau)))
    H2MixedPacketMatvecPlan(engine, precision)
end
Base.size(p::H2MixedPacketMatvecPlan) = p.engine.shape

const TransposedMixedH2Plan = Union{Transpose{Float64,H2MixedPacketMatvecPlan},Adjoint{Float64,H2MixedPacketMatvecPlan}}
LinearAlgebra.mul!(y::AbstractVector, p::H2MixedPacketMatvecPlan, x::AbstractVector, alpha::Number=1, beta::Number=0) =
    _packet_mul!(y, p.engine, x, alpha, beta, false)
LinearAlgebra.mul!(y::AbstractVector, p::TransposedMixedH2Plan, x::AbstractVector, alpha::Number=1, beta::Number=0) =
    _packet_mul!(y, parent(p).engine, x, alpha, beta, true)
LinearAlgebra.mul!(Y::AbstractMatrix, p::H2MixedPacketMatvecPlan, X::AbstractMatrix, alpha::Number=1, beta::Number=0) =
    _packet_mul_k!(Y, p.engine, X, alpha, beta, false)
LinearAlgebra.mul!(Y::AbstractMatrix, p::TransposedMixedH2Plan, X::AbstractMatrix, alpha::Number=1, beta::Number=0) =
    _packet_mul_k!(Y, parent(p).engine, X, alpha, beta, true)
Base.:*(p::Union{H2MixedPacketMatvecPlan,TransposedMixedH2Plan}, x::AbstractVector) = mul!(zeros(size(p, 1)), p, x)
Base.:*(p::Union{H2MixedPacketMatvecPlan,TransposedMixedH2Plan}, X::AbstractMatrix) = mul!(zeros(size(p, 1), size(X, 2)), p, X)
storage_bytes(p::H2MixedPacketMatvecPlan) = storage_bytes(p.engine)
multi_workspace_bytes(p::H2MixedPacketMatvecPlan, k::Integer) = multi_workspace_bytes(p.engine, k)
multi_workspace_bytes(p::H2MixedPacketMatvecPlan) = multi_workspace_bytes(p.engine)
release_multi_workspace!(p::H2MixedPacketMatvecPlan) = release_multi_workspace!(p.engine)
Base.copy(p::H2MixedPacketMatvecPlan) = H2MixedPacketMatvecPlan(copy(p.engine), p.precision)
"""
    precision_summary(plan::H2MixedPacketMatvecPlan)

Selection and error bounds of a mixed-precision plan: `rtol` and
`reference_norm` (η), the a priori Frobenius bound `bound` (≤ rtol·η), the
exactly computed Frobenius norm of the reduced-precision rounding of the stored
rows (`storage_perturbation`, scaled by basis norms), the numbers of Float32
and 48-bit rows, rotated and absorbed packets, packets excluded from Float32
by the overflow guard, and byte counts.
"""
precision_summary(p::H2MixedPacketMatvecPlan) = p.precision
Base.show(io::IO, p::H2MixedPacketMatvecPlan) = print(io, "H2MixedPacketMatvecPlan(", size(p, 1), " × ", size(p, 2), ", ",
    p.engine.workers, " workers, ", storage_bytes(p), " numeric bytes, precision_rtol=", p.precision.rtol, ")")
Base.show(io::IO, ::MIME"text/plain", p::H2MixedPacketMatvecPlan) = show(io, p)
