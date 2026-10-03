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
# The packets run on the task engine of H2PacketMatvecPlan (phases, write
# ownership, slot reductions, near-field column packets) and every row block
# is stored column-major and read by the engine's fused kernels, which widen
# reduced-precision entries on load. Products are deterministic and bitwise
# independent of the worker count; without reduced rows they equal the packet
# plan's bitwise.

const _U32 = Float64(eps(Float32)) / 2      # unit roundoff of Float32 rounding to nearest

const _U48 = 2.0^-37                        # unit roundoff of the 48-bit format below

# Optional 48-bit format: a Float64 rounded to sign, exponent and 36 mantissa
# bits (round to nearest), stored column-major as a UInt32 plane and a UInt16
# plane. Linear indexing decodes an entry, so the packet kernels read it directly.
struct _Matrix48 <: AbstractMatrix{Float64}
    hi::Vector{UInt32}
    lo::Vector{UInt16}
    m::Int
    n::Int
end
_Matrix48(::UndefInitializer, m::Int, n::Int) = _Matrix48(Vector{UInt32}(undef, m * n), Vector{UInt16}(undef, m * n), m, n)
Base.size(A::_Matrix48) = (A.m, A.n)
Base.IndexStyle(::Type{_Matrix48}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(A::_Matrix48, q::Int) =
    reinterpret(Float64, (UInt64(A.hi[q]) << 32) | (UInt64(A.lo[q]) << 16))
Base.@propagate_inbounds function Base.setindex!(A::_Matrix48, v, q::Int)
    u = (reinterpret(UInt64, Float64(v)) + 0x0000_0000_0000_8000) & 0xffff_ffff_ffff_0000
    A.hi[q] = UInt32(u >> 32); A.lo[q] = UInt16((u >> 16) & 0xffff)
    v
end
_nbytes(A::Matrix) = sizeof(A)
_nbytes(A::_Matrix48) = sizeof(A.hi) + sizeof(A.lo)
# Largest packet Frobenius norm whose rows may be stored in each reduced format:
# every stored entry is bounded by the norm (up to rotation roundoff), so the
# factors keep rounding (and the 48-bit carry) away from overflow.
const _MAX32 = Float64(floatmax(Float32)) / 2
const _MAX48 = floatmax(Float64) / 4
# Store A into columns off+1:off+size(A,2) of S (rounding to S's format) and
# return the squared Frobenius norm of the rounding error, computed exactly
# from the stored values.
function _store_block!(S::AbstractMatrix, A::AbstractMatrix, off::Int)
    e = 0.0
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        S[i, off+j] = A[i, j]
        e += abs2(Float64(S[i, off+j]) - A[i, j])
    end
    e
end

# Coupling packet of the mixed plan: rows 1:size(hi,1) in Float64, then
# 48-bit, then Float32 rows, all over the packet's N columns (colcoeff segments
# `columns`, adjoint slots `slot+1:slot+N`), each block column-major as in
# `_CouplingPacket` (which it equals bitwise without reduced rows).
struct _MixedCouplingPacket
    row::UnitRange{Int}
    columns::Vector{UnitRange{Int}}
    slot::Int
    hi::Matrix{Float64}
    mid::_Matrix48
    lo::Matrix{Float32}
    # Explicit packet rotation Q = H_1 ⋯ H_r as Householder reflectors (empty if
    # none or absorbed into the row basis): reflector i is [1; hv[off_i+1:off_i+k-i]].
    hv::Vector{Float64}
    tau::Vector{Float64}
end
_packet_ncols(b::_MixedCouplingPacket) = size(b.hi, 2)
_packet_bytes(b::_MixedCouplingPacket) = sizeof(b.hi) + _nbytes(b.mid) + sizeof(b.lo) + sizeof(b.hv) + sizeof(b.tau)
# Time model for task ordering (equals the packet plan's without reduced rows):
# Float32 elements stream about 1.4x faster than Float64 ones.
_packet_work(b::_MixedCouplingPacket) = length(b.mid) + length(b.lo) + length(b.tau) == 0 ? Float64(length(b.hi)) :
    length(b.hi) + 0.8 * length(b.mid) + 0.7 * length(b.lo) + 4.0 * length(b.hv)
# Per-worker scratch (per right-hand side): rotated rows.
_packet_scratch(b::_MixedCouplingPacket) = isempty(b.tau) ? 0 : length(b.row)
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

# y[y0+1:y0+m] += M * x[columns] (M: one row block of a packet; empty blocks skipped).
function _segments_n!(y, y0, M::AbstractMatrix, columns, x)
    size(M, 1) == 0 && return nothing
    off = 0
    for cr in columns
        _kernel_n!(y, y0, M, off, length(cr), x, first(cr) - 1); off += length(cr)
    end
    nothing
end
function _segments_nk!(Y, y0, ly, M::AbstractMatrix, columns, X, lx, K)
    size(M, 1) == 0 && return nothing
    off = 0
    for cr in columns
        _kernel_nk!(Y, y0, ly, M, off, length(cr), X, first(cr) - 1, lx, K); off += length(cr)
    end
    nothing
end
# Forward: rowcoeff[row] += Q W colcoeff[columns]; adjoint: slot = Wᵀ Qᵀ rowcoeff[row].
# Without a stored rotation these are exactly the `_CouplingPacket` operations.
function _coupling_task!(p, b::_MixedCouplingPacket, w, t)
    s = p.scratch[w]; N = size(b.hi, 2); k = length(b.row); r0 = first(b.row) - 1
    o2 = size(b.hi, 1); o3 = o2 + size(b.mid, 1)
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
        size(b.hi, 1) > 0 && _kernel_t!(sl, s0, b.hi, 0, N, src, wo)
        size(b.mid, 1) > 0 && _kernel_t!(sl, s0, b.mid, 0, N, src, wo + o2)
        size(b.lo, 1) > 0 && _kernel_t!(sl, s0, b.lo, 0, N, src, wo + o3)
    else
        if isempty(b.tau)
            z = p.rowcoeff; zo = r0
        else
            z = s; zo = 0
            @inbounds for i in 1:k; s[i] = 0.0; end
        end
        _segments_n!(z, zo, b.hi, b.columns, p.colcoeff)
        _segments_n!(z, zo + o2, b.mid, b.columns, p.colcoeff)
        _segments_n!(z, zo + o3, b.lo, b.columns, p.colcoeff)
        if !isempty(b.tau)
            _apply_reflectors!(s, 0, k, b.hv, b.tau, false)
            y = p.rowcoeff
            @inbounds for i in 1:k; y[r0+i] += s[i]; end
        end
    end
    nothing
end
function _coupling_task_k!(p, ws, b::_MixedCouplingPacket, w, t, K)
    lrc = length(p.rowcoeff); lcc = length(p.colcoeff); ls = length(p.slots)
    s = ws.scratch[w]; N = size(b.hi, 2); k = length(b.row); r0 = first(b.row) - 1
    o2 = size(b.hi, 1); o3 = o2 + size(b.mid, 1)
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
        size(b.hi, 1) > 0 && _kernel_tk!(sl, s0, ls, b.hi, 0, N, src, wo, lw, K)
        size(b.mid, 1) > 0 && _kernel_tk!(sl, s0, ls, b.mid, 0, N, src, wo + o2, lw, K)
        size(b.lo, 1) > 0 && _kernel_tk!(sl, s0, ls, b.lo, 0, N, src, wo + o3, lw, K)
    else
        if isempty(b.tau)
            z = ws.rowcoeff; zo = r0; lz = lrc
        else
            z = s; zo = 0; lz = k
            @inbounds for i in 1:k*K; s[i] = 0.0; end
        end
        _segments_nk!(z, zo, lz, b.hi, b.columns, ws.colcoeff, lcc, K)
        _segments_nk!(z, zo + o2, lz, b.mid, b.columns, ws.colcoeff, lcc, K)
        _segments_nk!(z, zo + o3, lz, b.lo, b.columns, ws.colcoeff, lcc, K)
        if !isempty(b.tau)
            y = ws.rowcoeff
            for v in 0:K-1
                _apply_reflectors!(s, v * k, k, b.hv, b.tau, false)
                @inbounds for i in 1:k; y[r0+i+v*lrc] += s[v*k+i]; end
            end
        end
    end
    nothing
end

"""
    H2MixedPacketMatvecPlan(compact_plan; workers=1, precision_rtol=1e-13, format48=false)
    H2MixedPacketMatvecPlan(h2; workers=1, precision_rtol=1e-13, format48=false, consume=false,
                            compact_options...)

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
small weight are stored in Float32 and the leading rows in Float64, each row
block column-major and read by the fused kernels of `H2PacketMatvecPlan`,
which widen reduced-precision entries on load. Every product accumulates in Float64; Float32 values are only
widened, never used as accumulators. Rotations of explicit (unsaturated) row
bases are absorbed into copies of the transfers and leaf bases, so they cost
no storage. For implicit (saturated) or physical rows, and for pass-through nodes and their children (whose
coefficients are copied between parent and children without a transfer), `Q`
is stored as the `r` Householder reflectors spanning the rows above the
lowest-precision block (about `8r(k - r/2)` bytes). That cost enters the
selection. The rounding cost
depends only on these subspaces, not on the basis chosen inside them.

With `format48=true`, rows of intermediate weight can also use a 48-bit format:
a Float64 rounded to 36 mantissa bits (unit roundoff `u₄₈ = 2⁻³⁷`), stored as
a UInt32 plane plus a UInt16 plane. This lowers storage further, but its
decode costs time, so products are usually slower than with Float32/Float64
rows only.

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
stored entirely in Float64 (and Float64 subnormal entries, below about
`2e-308`, are not covered by the 48-bit relative rounding bound).
`precision_rtol=0` keeps everything in Float64; its products are then bitwise
identical to `H2PacketMatvecPlan`. Near-field blocks always stay in Float64
(their spectra are flat, so rotating them saves almost nothing).

Forward and adjoint products apply the same stored values, so
`adjoint(plan)` is the exact transpose of the stored mixed operator.
Construction is bitwise deterministic for any number of Julia threads.
`consume=true` (H2 input only) releases the source operator's blocks while
the packets are built, as in `H2PacketMatvecPlan`; the source `h2` is
unusable afterwards.

From an H2 matrix, `compact_options` (`passthrough`, `coupling_rtol`,
`coupling_scale`) are passed to [`H2CompactMatvecPlan`](@ref), and `A` is then
that compact plan's operator. `passthrough=true` composes exactly (it only
changes the coefficient coordinates). Couplings factorized by `coupling_rtol`
are multiplied out here (`L*R'`), so coupling truncation does not reduce this
plan's storage; [`H2PacketMatvecPlan`](@ref) keeps such factors. A compact plan
with `coupling_precision` other than Float64 already stores reduced-precision
couplings and is rejected (`ArgumentError`, raised before anything is
consumed). `precision_summary(plan)` reports the selection and both
bounds.
"""
struct H2MixedPacketMatvecPlan <: AbstractMatrix{Float64}
    engine::H2PacketMatvecPlan{_MixedCouplingPacket}
    precision::NamedTuple
end
function H2MixedPacketMatvecPlan(h::H2Matrix; workers::Int=1, precision_rtol::Real=1e-13, format48::Bool=false,
                                 consume::Bool=false, compact_options...)
    # Validate before anything is consumed.
    _validate_mixed_options(workers, precision_rtol)
    get(compact_options, :coupling_precision, Float64) === Float64 || throw(_MIXED_COUPLING_PRECISION_ERROR)
    core = H2CompactMatvecPlan(h; consume, compact_options...)
    # The compact plan references every numerical block it needs (see H2PacketMatvecPlan).
    consume && _release_h2_blocks!(h)
    H2MixedPacketMatvecPlan(core; workers, precision_rtol, format48, _release=consume)
end
function _validate_mixed_options(workers, precision_rtol)
    workers > 0 || throw(ArgumentError("workers must be positive"))
    isfinite(precision_rtol) && precision_rtol >= 0 || throw(ArgumentError("precision_rtol must be finite and nonnegative"))
    nothing
end
const _MIXED_COUPLING_PRECISION_ERROR = ArgumentError("H2MixedPacketMatvecPlan needs Float64 couplings: " *
    "coupling_precision stores reduced-precision couplings by itself (use H2PacketMatvecPlan for it)")
# Float64 coupling blocks of a packet (factorized couplings are multiplied out).
_coupling_matrix(b::_PlanCoupling) = b.S
_coupling_matrix(b::_LowRankPlanCoupling) = b.R === nothing ? b.L : b.L * b.R'
_with_basis(n::_CompactBasisNode, V, E) = _CompactBasisNode(V, E, n.coeff, n.indices, n.children, n.identity, n.passthrough)

# Spectral norm of each compact basis node's full (nested) basis, via Gram matrices.
# A pass-through node's basis is block diagonal in its children's bases.
function _compact_basis_norms(nodes::Vector{_CompactBasisNode})
    G = Vector{Matrix{Float64}}(undef, length(nodes)); nrm = ones(length(nodes))
    for i in reverse(eachindex(nodes))
        n = nodes[i]; k = length(n.coeff)
        if n.identity || k == 0
            G[i] = zeros(0, 0)
            continue
        elseif isempty(n.children)
            G[i] = n.V' * n.V
        elseif n.passthrough
            g = zeros(k, k); off = 0
            for j in n.children
                c = nodes[j]; kc = length(c.coeff)
                kc == 0 && continue
                block = view(g, off+1:off+kc, off+1:off+kc)
                c.identity ? copyto!(block, I(kc)) : copyto!(block, G[j])
                off += kc
            end
            G[i] = g
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
    _validate_mixed_options(workers, precision_rtol)
    p isa H2CompactMatvecPlan{_MixedPlanCoupling} && throw(_MIXED_COUPLING_PRECISION_ERROR)
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
    # Rotations of explicit row bases are absorbed; implicit (saturated) rows
    # store U, and so do pass-through nodes and their children, whose
    # coefficients are copied between parent and children without a transfer.
    sharedcoeff = [n.identity || n.passthrough for n in p.rows]
    for n in p.rows
        n.passthrough && (sharedcoeff[n.children] .= true)
    end
    explicitU = [sharedcoeff[order[t]] for t in 1:nf]
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
            rows[i] = _with_basis(n, V, E)
            leftdone[i] = true
            for j in n.children
                c = rows[j]
                isempty(c.E) && continue
                if willabsorb[j] && !leftdone[j]
                    heldright[j] = U
                else
                    rows[j] = _with_basis(c, c.V, c.E * U)
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
        hirows = Matrix{Float64}(undef, r1, N); midrows = _Matrix48(undef, rm - r1, N); lorows = Matrix{Float32}(undef, k - rm, N)
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
            copyto!(view(hirows, :, off+1:off+size(B, 2)), hiW)
            c += _U32^2 * sum(abs2, loW) + length(loW) * 2.0^-300 + _U48^2 * sum(abs2, midW)
            e += _store_block!(lorows, loW, off) + _store_block!(midrows, midW, off)
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
        float32_bytes=sel(b -> sizeof(b.lo)), float48_bytes=sel(b -> _nbytes(b.mid)),
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
