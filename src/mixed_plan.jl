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

# Row-interleaved storage of a dense `rows x N` matrix: groups of four rows are
# stored column-interleaved (`data[4N*g+4(j-1)+q] = A[4g+q,j]`), and the trailing
# `rows % 4` rows follow, each contiguous.
struct _InterleavedRows{T<:Union{Float32,Float64}}
    data::Vector{T}
    rows::Int
    N::Int
end
function _InterleavedRows{T}(A::AbstractMatrix) where {T}
    rows, N = size(A); data = Vector{T}(undef, rows * N); g4 = rows ÷ 4
    @inbounds for g in 0:g4-1, j in 1:N, q in 1:4
        data[4N*g+4(j-1)+q] = T(A[4g+q, j])
    end
    @inbounds for r in 1:rows%4, j in 1:N
        data[4N*g4+N*(r-1)+j] = T(A[4g4+r, j])
    end
    _InterleavedRows{T}(data, rows, N)
end
function Base.Matrix(A::_InterleavedRows{T}) where {T}
    M = Matrix{T}(undef, A.rows, A.N); N = A.N; g4 = A.rows ÷ 4
    for g in 0:g4-1, j in 1:N, q in 1:4
        M[4g+q, j] = A.data[4N*g+4(j-1)+q]
    end
    for r in 1:A.rows%4, j in 1:N
        M[4g4+r, j] = A.data[4N*g4+N*(r-1)+j]
    end
    M
end
@inline function _il_fwd4!(z, zo, d::Vector{T}, o, N, x) where {T}
    s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0
    @inbounds @simd for j in 1:N
        xj = x[j]; q = o + 4(j - 1)
        s1 = muladd(Float64(d[q+1]), xj, s1); s2 = muladd(Float64(d[q+2]), xj, s2)
        s3 = muladd(Float64(d[q+3]), xj, s3); s4 = muladd(Float64(d[q+4]), xj, s4)
    end
    @inbounds begin
        z[zo+1] += s1; z[zo+2] += s2; z[zo+3] += s3; z[zo+4] += s4
    end
    nothing
end
@inline function _il_fwd1!(z, zo, d::Vector{T}, o, N, x) where {T}
    s = 0.0
    @inbounds @simd for j in 1:N
        s = muladd(Float64(d[o+j]), x[j], s)
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
@inline function _il_adj4!(x, d::Vector{T}, o, N, w, wo) where {T}
    @inbounds w1 = w[wo+1]; @inbounds w2 = w[wo+2]; @inbounds w3 = w[wo+3]; @inbounds w4 = w[wo+4]
    @inbounds @simd for j in 1:N
        q = o + 4(j - 1)
        x[j] += muladd(Float64(d[q+1]), w1, Float64(d[q+2]) * w2) + muladd(Float64(d[q+3]), w3, Float64(d[q+4]) * w4)
    end
    nothing
end
@inline function _il_adj1!(x, d::Vector{T}, o, N, wi) where {T}
    @inbounds @simd for j in 1:N
        x[j] = muladd(Float64(d[o+j]), wi, x[j])
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
    hi::_InterleavedRows{Float64}
    lo::_InterleavedRows{Float32}
    # Explicit packet rotation Q = H_1 ⋯ H_r as Householder reflectors (empty if
    # none or absorbed into the row basis): reflector i is [1; hv[off_i+1:off_i+k-i]].
    hv::Vector{Float64}
    tau::Vector{Float64}
    scratch::Vector{Float64}
    zbuf::Vector{Float64}
end
_copy_packet(b::_MixedPacket) = _MixedPacket(b.row, b.columns, b.hi, b.lo, b.hv, b.tau, zeros(length(b.scratch)), zeros(length(b.zbuf)))
_packet_bytes(b::_MixedPacket) = sizeof(b.hi.data) + sizeof(b.lo.data) + sizeof(b.hv) + sizeof(b.tau)
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
    H2MixedPacketMatvecPlan(compact_plan; workers=1, precision_rtol=1e-13)
    H2MixedPacketMatvecPlan(h2; workers=1, precision_rtol=1e-13)

Packet plan with adaptive mixed-precision storage. Far-field packets hold all
couplings of one row node, as in `H2PacketMatvecPlan`. Near-field packets hold
all dense blocks of one leaf row range.

Each packet `P` (`k × N`) is rotated to its left singular basis, `P = Q W`.
Row `i` of `W` then has norm `ω_i = σ_i(P)`. Trailing rows with small weight are
stored in Float32 and the leading rows in Float64. Every product accumulates
in Float64; Float32 values are only widened, never used as accumulators.
Rotations of explicit (unsaturated) row bases are absorbed exactly into the
stored transfers and leaf bases, so they cost no storage. For implicit
(saturated) or physical rows, `Q` is stored as the `k - r` Householder
reflectors that span the `r` Float64 rows. That cost enters the selection.

A global Lagrangian allocation picks the rows. It minimizes stored bytes
subject to the a priori bound

    ‖Ã - A‖₂ ≤ ‖Ã - A‖_F ≤ (Σ_t β_t² Σ_{i∈lo(t)} (u₃₂² ω_i² + N_t 2⁻³⁰⁰))^{1/2} ≤ precision_rtol · η.

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

# Bytes saved and bound cost for storing the r2 trailing rows of a packet in Float32.
# Explicit rotations store the reflectors spanning the r1 = k-r2 Float64 rows;
# any orthonormal basis of their complement gives the same Float32 cost.
function _best_split(om2::Vector{Float64}, N::Int, explicit::Bool, beta2::Float64, mu::Float64)
    k = length(om2); best = 0.0; bestr = 0; c = 0.0; bestc = 0.0
    for r2 in 1:k
        c += beta2 * (_U32^2 * om2[k-r2+1] + N * 2.0^-300)
        ucost = explicit ? 8.0 * (_reflector_entries(k, k - r2) + (k - r2)) : 0.0
        f = 4.0 * r2 * N - ucost - mu * c
        if f > best
            best = f; bestr = r2; bestc = c
        end
    end
    bestr, bestc
end
# Smallest Lagrange multiplier whose per-packet selections meet the squared budget.
function _precision_selection(om2, Ns, explicit, beta2, delta2)
    np = length(om2); r2 = zeros(Int, np)
    delta2 > 0 || return r2
    total(mu) = sum(_best_split(om2[t], Ns[t], explicit[t], beta2[t], mu)[2] for t in 1:np; init=0.0)
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
        r2[t] = _best_split(om2[t], Ns[t], explicit[t], beta2[t], mu)[1]
    end
    r2
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

function H2MixedPacketMatvecPlan(p::H2CompactMatvecPlan; workers::Int=1, precision_rtol::Real=1e-13)
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
    # Packets 1:nf are far-field (one per row node), nf+1:np near-field (one per leaf row range).
    packet_matrix(t) = t <= nf ? reduce(hcat, [_coupling_matrix(p.couplings[i]) for i in groups[order[t]]]) :
        reduce(hcat, [p.dense[i].D[rr, :] for (i, rr) in neargroups[nearorder[t-nf]]])
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
    # Pass 1: rotation and row weights of every packet (packet matrices are temporary).
    Us = Vector{Matrix{Float64}}(undef, np); om2 = [Float64[] for _ in 1:np]; fro2 = zeros(np)
    rot = precision_rtol > 0
    Threads.@threads :dynamic for t in 1:np
        M = packet_matrix(t); fro2[t] = sum(abs2, M)
        if rot
            # Eigenvectors of the Gram matrix give the left singular basis; only
            # the selection uses these weights, the bound uses the stored rows.
            F = eigen(Symmetric(M * M'); alg=LinearAlgebra.DivideAndConquer())
            Us[t] = F.vectors[:, end:-1:1]; om2[t] = max.(F.values[end:-1:1], 0.0)
        end
    end
    eta = sqrt(sum(fro2; init=0.0))
    r2 = rot ? _precision_selection(om2, Ns, explicitU, beta2, (precision_rtol * eta)^2) : zeros(Int, np)
    # A packet stored entirely in Float32 needs no rotation.
    rotated = [0 < r2[t] < ks[t] for t in 1:np]
    for t in 1:np
        rotated[t] || (Us[t] = zeros(0, 0))     # release unused pass-1 rotations early
    end
    absorbed = Dict{Int,Matrix{Float64}}(order[t] => Us[t] for t in 1:nf if rotated[t] && !explicitU[t])
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
    # Pass 2: rotated packets split into Float64 and Float32 rows.
    packets = Vector{_MixedPacket}(undef, np)
    lo2 = zeros(np); err2 = zeros(np)
    Threads.@threads :dynamic for t in 1:np
        M = packet_matrix(t); k = ks[t]; r1 = k - r2[t]
        hv = Float64[]; tau = Float64[]
        W = if !rotated[t]
            M
        elseif !explicitU[t]
            Us[t]' * M
        else
            # Householder QR of the leading r1 singular vectors: Q[:, 1:r1] spans
            # them and Q[:, r1+1:k] their orthogonal complement.
            A, tau = LAPACK.geqrf!(Us[t][:, 1:r1])
            Q = LAPACK.ormqr!('L', 'N', A, tau, Matrix{Float64}(I, k, k))
            hv = reduce(vcat, [A[i+1:k, i] for i in 1:r1])
            Q' * M
        end
        hiW = view(W, 1:r1, :); loW = view(W, r1+1:k, :)
        lorows = _InterleavedRows{Float32}(loW)
        lo2[t] = beta2[t] * (_U32^2 * sum(abs2, loW) + length(loW) * 2.0^-300)
        err2[t] = beta2[t] * sum(abs2, Float64.(Matrix(lorows)) .- loW)
        packets[t] = _MixedPacket(outrows[t], columns[t], _InterleavedRows{Float64}(hiW), lorows,
            hv, tau, zeros(Ns[t]), zeros(isempty(tau) ? 0 : k))
    end
    # Balanced static schedule over far and near packets (time model: Float32
    # elements stream about 1.4x faster than Float64 elements).
    cost = [b.hi.rows * b.hi.N + 0.7 * b.lo.rows * b.lo.N + 4.0 * length(b.hv) + 2.0 * b.hi.N + 256.0 for b in packets]
    owner = _lpt_schedule(cost, workers)
    farschedule = [[t for t in 1:nf if owner[t] == w] for w in 1:workers]
    nearschedule = [[t - nf for t in nf+1:np if owner[t] == w] for w in 1:workers]
    sel(f, range) = sum((f(packets[t]) for t in range); init=0)
    precision = (; rtol=Float64(precision_rtol), reference_norm=eta,
        bound=sqrt(sum(lo2; init=0.0)), storage_perturbation=sqrt(sum(err2; init=0.0)),
        float32_rows=sum(r2; init=0), rows=sum(ks; init=0), rotated_packets=count(rotated), packets=np,
        near_rotated_packets=count(view(rotated, nf+1:np)), near_float32_rows=sum(view(r2, nf+1:np); init=0), absorbed_rotations=length(absorbed),
        float64_far_bytes=sum((8 * ks[t] * Ns[t] for t in 1:nf); init=0),
        float64_near_bytes=sum((8 * ks[t] * Ns[t] for t in nf+1:np); init=0),
        far_bytes=sel(_packet_bytes, 1:nf), near_bytes=sel(_packet_bytes, nf+1:np),
        float32_bytes=sel(b -> sizeof(b.lo.data), 1:np), rotation_bytes=sel(b -> sizeof(b.hv) + sizeof(b.tau), 1:np))
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
        _il_forward!(rowcoeff, zo + b.hi.rows, b.lo, s)
    else
        z = b.zbuf; fill!(z, 0.0)
        _il_forward!(z, 0, b.hi, s)
        _il_forward!(z, b.hi.rows, b.lo, s)
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
        _il_adjoint!(s, b.lo, rowcoeff, wo + b.hi.rows)
    else
        z = b.zbuf; zo = first(b.row) - 1
        @inbounds for i in eachindex(z)
            z[i] = rowcoeff[zo+i]
        end
        _apply_reflectors!(z, b.hv, b.tau, true)
        _il_adjoint!(s, b.hi, z, 0)
        _il_adjoint!(s, b.lo, z, b.hi.rows)
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
