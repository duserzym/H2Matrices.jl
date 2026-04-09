"""
    H²-Matrix–Vector Product (O(N) complexity)

The H² matvec `y = A * x` is performed in three phases:

1. **Forward (upward) pass**: Project `x` onto the column cluster basis,
   computing coefficient vectors `x̂_σ = W_σ' * x|_σ` from leaves to root
   using transfer matrices.

2. **Interaction (coupling)**: For each admissible block (τ, σ), compute
   `ŷ_τ += S_{τσ} * x̂_σ` using the small coupling matrices.

3. **Backward (downward) pass**: Expand coefficient vectors `ŷ_τ` back to
   DOF space via the row cluster basis, from root to leaves:
   `y|_τ += V_τ * ŷ_τ`.

The near-field (inadmissible) blocks are handled by direct dense matvec.
"""

# ──────────────────────────────────────────────────────────────────
# Coefficient vectors stored per cluster node
# ──────────────────────────────────────────────────────────────────

"""
    allocate_coefficients(cb::ClusterBasis)

Allocate a dictionary mapping each cluster basis node to a zero coefficient
vector of length `cb.k`.
"""
function allocate_coefficients(cb::ClusterBasis{N,T}) where {N,T}
    coeffs = Dict{ClusterBasis{N,T}, Vector{Float64}}()
    _alloc_coeffs!(coeffs, cb)
    return coeffs
end

function _alloc_coeffs!(coeffs, cb::ClusterBasis{N,T}) where {N,T}
    coeffs[cb] = zeros(Float64, cb.k)
    for child in cb.children
        _alloc_coeffs!(coeffs, child)
    end
    return coeffs
end

# ──────────────────────────────────────────────────────────────────
# Phase 1: Forward (upward) transform
# ──────────────────────────────────────────────────────────────────

"""
    forward_transform!(coeffs, cb, x)

Compute the forward (upward) transformation:
- At each leaf: `x̂_t = V_t' * x|_t`
- At each non-leaf: `x̂_t = Σ_i E_i' * x̂_{t_i}` (sum over children)

After this call, `coeffs[cb]` contains the projected coefficients for
every node in the cluster basis tree.

# Arguments
- `coeffs` : dictionary of coefficient vectors (output)
- `cb` : the column cluster basis
- `x` : the input vector (in local ordering)
"""
function forward_transform!(
    coeffs::Dict{ClusterBasis{N,T}, Vector{Float64}},
    cb::ClusterBasis{N,T},
    x::AbstractVector
) where {N,T}
    irange = index_range(cb.cluster)
    root_start = index_range(_root_cluster(cb)).start

    if isleaf(cb)
        # x̂_t = V_t' * x|_t
        local_range = irange .- (root_start - 1)
        x_local = view(x, local_range)
        if cb.k > 0
            mul!(coeffs[cb], cb.V', x_local)
        end
    else
        # Recurse on children first (bottom-up)
        for child in cb.children
            forward_transform!(coeffs, child, x)
        end
        # x̂_t = Σ_i E_i' * x̂_{t_i}
        if cb.k > 0
            fill!(coeffs[cb], 0.0)
            for child in cb.children
                if child.k > 0 && size(child.E, 2) > 0
                    # E is k_child × k_parent, so E' is k_parent × k_child
                    mul!(coeffs[cb], child.E', coeffs[child], 1.0, 1.0)
                end
            end
        end
    end
    return coeffs
end

# ──────────────────────────────────────────────────────────────────
# Phase 2: Interaction (coupling)
# ──────────────────────────────────────────────────────────────────

"""
    interaction!(row_coeffs, h2, col_coeffs)

For each admissible leaf block, compute:
    ŷ_τ += S_{τσ} * x̂_σ
"""
function interaction!(
    row_coeffs::Dict{ClusterBasis{N,T}, Vector{Float64}},
    h2::H2Matrix{N,T},
    col_coeffs::Dict{ClusterBasis{N,T}, Vector{Float64}}
) where {N,T}
    if isleaf(h2) && isadmissible(h2) && h2.uniform !== nothing
        rb = h2.row_basis
        cb_col = h2.col_basis
        S = h2.uniform.S
        if rb.k > 0 && cb_col.k > 0
            mul!(row_coeffs[rb], S, col_coeffs[cb_col], 1.0, 1.0)
        end
    elseif !isleaf(h2)
        for child in h2.children
            interaction!(row_coeffs, child, col_coeffs)
        end
    end
    return row_coeffs
end

# ──────────────────────────────────────────────────────────────────
# Phase 3: Backward (downward) transform
# ──────────────────────────────────────────────────────────────────

"""
    backward_transform!(y, coeffs, cb)

Compute the backward (downward) transformation:
- At root: start with accumulated coefficient ŷ_root
- At each non-leaf: propagate `ŷ_{t_i} += E_i * ŷ_t` to children
- At each leaf: `y|_t += V_t * ŷ_t`

# Arguments
- `y` : output vector (in local ordering, modified in-place)
- `coeffs` : dictionary of coefficient vectors (input)
- `cb` : the row cluster basis
"""
function backward_transform!(
    y::AbstractVector,
    coeffs::Dict{ClusterBasis{N,T}, Vector{Float64}},
    cb::ClusterBasis{N,T}
) where {N,T}
    irange = index_range(cb.cluster)
    root_start = index_range(_root_cluster(cb)).start

    if isleaf(cb)
        # y|_t += V_t * ŷ_t
        local_range = irange .- (root_start - 1)
        if cb.k > 0
            mul!(view(y, local_range), cb.V, coeffs[cb], 1.0, 1.0)
        end
    else
        # Propagate down: ŷ_{t_i} += E_i * ŷ_t
        for child in cb.children
            if child.k > 0 && cb.k > 0 && size(child.E, 2) > 0
                mul!(coeffs[child], child.E, coeffs[cb], 1.0, 1.0)
            end
        end
        # Recurse on children
        for child in cb.children
            backward_transform!(y, coeffs, child)
        end
    end
    return y
end

# ──────────────────────────────────────────────────────────────────
# Near-field (dense block) contribution
# ──────────────────────────────────────────────────────────────────

"""
    nearfield_matvec!(y, h2, x)

Add contributions from all inadmissible (dense) leaf blocks:
    y|_τ += D_{τσ} * x|_σ
"""
function nearfield_matvec!(
    y::AbstractVector,
    h2::H2Matrix{N,T},
    x::AbstractVector
) where {N,T}
    root_row_start = index_range(h2.row_basis.cluster).start
    root_col_start = index_range(h2.col_basis.cluster).start

    _nearfield_recursive!(y, h2, x, root_row_start - 1, root_col_start - 1)
    return y
end

function _nearfield_recursive!(y, h2::H2Matrix, x, row_offset, col_offset)
    if isleaf(h2)
        if !isadmissible(h2) && h2.dense !== nothing
            irange = index_range(h2.row_basis.cluster) .- row_offset
            jrange = index_range(h2.col_basis.cluster) .- col_offset
            mul!(view(y, irange), h2.dense, view(x, jrange), 1.0, 1.0)
        end
    else
        for child in h2.children
            _nearfield_recursive!(y, child, x, row_offset, col_offset)
        end
    end
    return y
end

# ──────────────────────────────────────────────────────────────────
# Full H² matrix-vector product
# ──────────────────────────────────────────────────────────────────

"""
    h2matvec!(y, h2, x; global_index=false)

Compute `y += A * x` where `A` is an H²-matrix.
Uses the three-phase algorithm: forward → interaction → backward + nearfield.

# Arguments
- `y` : output vector (modified in-place)
- `h2` : the H²-matrix
- `x` : input vector
- `global_index` : if true, x and y are in global indexing
"""
function h2matvec!(
    y::AbstractVector,
    h2::H2Matrix{N,T},
    x::AbstractVector;
    global_index::Bool=false
) where {N,T}
    rc = h2.row_basis.cluster
    cc = h2.col_basis.cluster

    if global_index
        # Permute to local ordering
        x_local = x[loc2glob(cc)]
        y_local = y[loc2glob(rc)]
    else
        x_local = x
        y_local = y
    end

    # Phase 1: Forward transform on column basis
    col_coeffs = allocate_coefficients(h2.col_basis)
    forward_transform!(col_coeffs, h2.col_basis, x_local)

    # Phase 2: Interaction (coupling)
    row_coeffs = allocate_coefficients(h2.row_basis)
    interaction!(row_coeffs, h2, col_coeffs)

    # Phase 3: Backward transform on row basis
    backward_transform!(y_local, row_coeffs, h2.row_basis)

    # Near-field contribution
    nearfield_matvec!(y_local, h2, x_local)

    if global_index
        invpermute!(y, loc2glob(rc))
    end
    return y
end

"""
    LinearAlgebra.mul!(y, h2::H2Matrix, x, α, β; global_index=false)

Standard mul! interface: `y = α * A * x + β * y`
"""
function LinearAlgebra.mul!(
    y::AbstractVector,
    h2::H2Matrix,
    x::AbstractVector,
    α::Number=1,
    β::Number=0;
    global_index::Bool=false
)
    # Scale y by β
    if iszero(β)
        fill!(y, zero(eltype(y)))
    elseif β != 1
        rmul!(y, β)
    end

    if iszero(α)
        return y
    end

    # Compute y += A * x
    if α == 1
        h2matvec!(y, h2, x; global_index)
    else
        # Use a temporary
        tmp = zeros(eltype(y), length(y))
        h2matvec!(tmp, h2, x; global_index)
        axpy!(α, tmp, y)
    end
    return y
end

Base.:*(h2::H2Matrix, x::AbstractVector) = mul!(zeros(size(h2, 1)), h2, x)

# ──────────────────────────────────────────────────────────────────
# Helper: find root cluster
# ──────────────────────────────────────────────────────────────────

function _root_cluster(cb::ClusterBasis)
    current = cb
    while !isroot(current)
        current = current.parent
    end
    return current.cluster
end
