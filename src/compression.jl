"""
    Adaptive H² Compression

Two main capabilities:

1. **H-matrix → H²-matrix conversion** (`compress_hmatrix_to_h2`):
   Build shared nested cluster bases from per-block ACA low-rank data,
   then compute coupling matrices by projection.

2. **H² recompression** (`recompress!`):
   Reduce basis ranks of an existing H²-matrix while maintaining
   error bounds, using weight-based SVD truncation.

Reference: Börm, "Efficient Numerical Methods for Non-local Operators",
           H2Lib (Börm et al.) — h2compression.c
"""

# ════════════════════════════════════════════════════════════════════
# Part 1: H-matrix → H²-matrix conversion
# ════════════════════════════════════════════════════════════════════

"""
    compress_hmatrix_to_h2(hmat; rtol=1e-8, maxrank=50)

Convert an assembled `HMatrix` (from HMatrices.jl) into an `H2Matrix`
with shared nested cluster bases.

The algorithm:
1. Collect per-block low-rank data (RkMatrix = A * B') from admissible leaves
2. Build row basis bottom-up: at leaves stack A columns → SVD → truncate;
   at non-leaves project through children → SVD → transfer matrices
3. Build column basis similarly using B columns
4. Mirror the H-matrix block tree as H²-matrix block tree
5. Compute coupling matrices by projecting RkMatrix data through the
   nested bases: S = (V_τ' A)(W_σ' B)'

# Arguments
- `hmat` : an assembled HMatrix from HMatrices.jl
- `rtol` : relative truncation tolerance for basis rank
- `maxrank` : maximum allowed rank per cluster

# Returns
An `H2Matrix` approximating the same kernel.
"""
function compress_hmatrix_to_h2(hmat::HMatrix;
                                rtol::Float64=1e-8,
                                maxrank::Int=50,
                                _print::Bool=true)
    rt = HMatrices.rowtree(hmat)
    ct = HMatrices.coltree(hmat)

    # Build cluster bases mirroring the cluster trees
    rb = build_cluster_basis(rt)
    cb = build_cluster_basis(ct)

    # Maps: objectid(ClusterTree node) → ClusterBasis node
    row_map = _build_obj_map(rb)
    col_map = _build_obj_map(cb)

    # Collect admissible block data grouped by row/col cluster
    row_data, col_data = _collect_rk_data(hmat)

    # Build row basis from A matrices (bottom-up with propagation)
    empty_inherited = Tuple{Matrix{Float64},UnitRange{Int}}[]
    _build_adaptive_basis_recursive!(rb, row_data, empty_inherited;
                                      rtol, maxrank, is_row=true)

    # Build col basis from B matrices (bottom-up with propagation)
    _build_adaptive_basis_recursive!(cb, col_data, Tuple{Matrix{Float64},UnitRange{Int}}[];
                                      rtol, maxrank, is_row=false)

    # Mirror H-matrix block tree as H² block tree
    h2 = _mirror_hmat_to_h2(hmat, row_map, col_map)

    # Fill coupling matrices and dense blocks from H-matrix data
    _fill_h2_from_hmat!(h2, hmat, row_map, col_map)

    h2.global_index = true
    _print && _print_compression_summary(h2)
    return h2
end

# ─────────────────────────────────────────────────────────────────
# Helpers for H-matrix → H² conversion
# ─────────────────────────────────────────────────────────────────

"""
Build a map from `objectid(clt_node)` → `ClusterBasis` node.
"""
function _build_obj_map(cb::ClusterBasis{N,T}) where {N,T}
    map = Dict{UInt,ClusterBasis{N,T}}()
    _fill_obj_map!(map, cb)
    return map
end

function _fill_obj_map!(map, cb::ClusterBasis{N,T}) where {N,T}
    map[objectid(cb.cluster)] = cb
    for child in cb.children
        _fill_obj_map!(map, child)
    end
end

"""
Collect admissible leaf block data from an H-matrix, grouped by cluster.

Returns:
- `row_data`: Dict mapping `objectid(row_cluster)` → Vector of (A=..., B=...)
- `col_data`: Dict mapping `objectid(col_cluster)` → Vector of (A=..., B=...)
"""
function _collect_rk_data(hmat::HMatrix)
    row_data = Dict{UInt,Vector{NamedTuple{(:A,:B),Tuple{Matrix{Float64},Matrix{Float64}}}}}()
    col_data = Dict{UInt,Vector{NamedTuple{(:A,:B),Tuple{Matrix{Float64},Matrix{Float64}}}}}()
    _collect_rk_recursive!(row_data, col_data, hmat)
    return row_data, col_data
end

function _collect_rk_recursive!(row_data, col_data, hmat::HMatrix)
    if HMatrices.isleaf(hmat)
        if HMatrices.isadmissible(hmat)
            d = HMatrices.data(hmat)
            if d !== nothing
                A = Matrix{Float64}(d.A)
                B = Matrix{Float64}(d.B)
                entry = (A=A, B=B)

                rt = HMatrices.rowtree(hmat)
                ct = HMatrices.coltree(hmat)
                rkey = objectid(rt)
                ckey = objectid(ct)

                push!(get!(row_data, rkey, valtype(row_data)()), entry)
                push!(get!(col_data, ckey, valtype(col_data)()), entry)
            end
        end
    else
        for child in HMatrices.children(hmat)
            _collect_rk_recursive!(row_data, col_data, child)
        end
    end
end

"""
Build adaptive basis bottom-up from collected RkMatrix data, with
top-down propagation of ancestor-level block data to leaf clusters.

For row basis: uses A columns; for col basis: uses B columns.
The `data` argument maps `objectid(cluster)` → Vector of (A=..., B=...).

Ancestor-level blocks are propagated down: their subrows are distributed
to child clusters so leaf bases capture all necessary column spaces.
"""
function _build_adaptive_basis_recursive!(
    cb::ClusterBasis{N,T},
    data::Dict,
    inherited::Vector{Tuple{Matrix{Float64},UnitRange{Int}}};
    rtol::Float64=1e-8,
    maxrank::Int=50,
    is_row::Bool=true
) where {N,T}
    my_range = index_range(cb.cluster)

    # 1. Collect direct blocks at this level
    my_direct_entries = get(data, objectid(cb.cluster), nothing)
    direct_matrices = Matrix{Float64}[]
    if my_direct_entries !== nothing
        for b in my_direct_entries
            push!(direct_matrices, is_row ? b.A : b.B)
        end
    end

    # 2. Extract subrows of inherited blocks for this cluster
    my_inherited_matrices = Matrix{Float64}[]
    for (M_anc, anc_range) in inherited
        local_start = my_range.start - anc_range.start + 1
        local_end = my_range.stop - anc_range.start + 1
        push!(my_inherited_matrices, M_anc[local_start:local_end, :])
    end

    # 3. Combine ALL data with rows matching this cluster
    all_active = vcat(direct_matrices, my_inherited_matrices)

    if isleaf(cb)
        m = length(cb)
        if isempty(all_active)
            cb.V = zeros(Float64, m, 0)
            cb.k = 0
        else
            C = hcat(all_active...)
            F = svd(C)
            k = _truncation_rank(F.S, rtol, maxrank)
            cb.V = F.U[:, 1:k]
            cb.k = k
        end
    else
        # 4. Pass all active data (direct + inherited) to children as inherited
        child_inherited = Tuple{Matrix{Float64},UnitRange{Int}}[]
        for m in direct_matrices
            push!(child_inherited, (m, my_range))
        end
        # Also forward the original inherited blocks (children will extract subrows)
        new_inherited = vcat(inherited, child_inherited)

        # 5. Recurse on children (bottom-up)
        for child in cb.children
            _build_adaptive_basis_recursive!(child, data, new_inherited;
                                              rtol, maxrank, is_row)
        end

        # 6. Build transfer matrices
        child_ks = [child.k for child in cb.children]
        total_child_k = sum(child_ks)

        if total_child_k == 0
            cb.k = 0
            for child in cb.children
                child.E = zeros(Float64, 0, 0)
            end
        elseif isempty(direct_matrices)
            # No direct blocks at this level → identity embedding
            # (inherited blocks are handled through children's bases already)
            _set_identity_embedding!(cb, child_ks, total_child_k)
        else
            # Project direct block data through children's bases
            irange = index_range(cb.cluster)
            projected_cols = Matrix{Float64}[]

            for M in direct_matrices
                proj = _project_through_children(cb, M, irange, child_ks, total_child_k)
                push!(projected_cols, proj)
            end

            C = hcat(projected_cols...)
            F = svd(C)
            k = _truncation_rank(F.S, rtol, maxrank)

            if k == 0
                _set_identity_embedding!(cb, child_ks, total_child_k)
            else
                cb.k = k
                U_k = F.U[:, 1:k]
                offset = 0
                for child in cb.children
                    child.E = U_k[(offset+1):(offset+child.k), :]
                    offset += child.k
                end
            end
        end
    end
    return cb
end

"""
Project a matrix M (with rows in parent range) through children's bases.
Returns a `(total_child_k × cols)` projected matrix.
"""
function _project_through_children(cb, M, parent_irange, child_ks, total_child_k)
    cols = size(M, 2)
    proj = zeros(Float64, total_child_k, cols)
    offset = 0
    for child in cb.children
        child_irange = index_range(child.cluster)
        local_start = child_irange.start - parent_irange.start + 1
        local_end = child_irange.stop - parent_irange.start + 1
        M_child = M[local_start:local_end, :]  # |child| × cols
        V_child = _full_basis(child)             # |child| × k_child
        proj[(offset+1):(offset+child.k), :] = V_child' * M_child
        offset += child.k
    end
    return proj
end

"""
Set identity embedding transfer matrices (no compression at parent level).
"""
function _set_identity_embedding!(cb, child_ks, total_child_k)
    cb.k = total_child_k
    offset = 0
    for child in cb.children
        E = zeros(Float64, child.k, total_child_k)
        for i in 1:child.k
            E[i, offset + i] = 1.0
        end
        child.E = E
        offset += child.k
    end
end

"""
Mirror an H-matrix block tree as an H²-matrix block tree.
"""
function _mirror_hmat_to_h2(hmat::HMatrix, row_map, col_map)
    rt = HMatrices.rowtree(hmat)
    ct = HMatrices.coltree(hmat)
    rb = row_map[objectid(rt)]
    cb = col_map[objectid(ct)]
    h2 = H2Matrix(rb, cb)
    _mirror_hmat_recursive!(h2, hmat, row_map, col_map)
    return h2
end

function _mirror_hmat_recursive!(h2::H2Matrix{N,T}, hmat::HMatrix,
                                  row_map, col_map) where {N,T}
    if HMatrices.isleaf(hmat)
        h2.admissible = HMatrices.isadmissible(hmat)
    else
        h2.admissible = false
        hchildren = HMatrices.children(hmat)
        nr, nc = size(hchildren)
        h2.children = Matrix{H2Matrix{N,T}}(undef, nr, nc)
        for i in 1:nr
            for j in 1:nc
                hc = hchildren[i, j]
                rtc = HMatrices.rowtree(hc)
                ctc = HMatrices.coltree(hc)
                rbc = row_map[objectid(rtc)]
                cbc = col_map[objectid(ctc)]
                child_h2 = H2Matrix(rbc, cbc)
                h2.children[i, j] = child_h2
                _mirror_hmat_recursive!(child_h2, hc, row_map, col_map)
            end
        end
    end
end

"""
Fill H²-matrix data from H-matrix block data.
- Admissible blocks: compute coupling matrix S = (V_row' A)(V_col' B)'
- Dense blocks: copy dense matrix directly
"""
function _fill_h2_from_hmat!(h2::H2Matrix{N,T}, hmat::HMatrix,
                              row_map, col_map) where {N,T}
    if HMatrices.isleaf(hmat) && isleaf(h2)
        if HMatrices.isadmissible(hmat)
            d = HMatrices.data(hmat)
            if d !== nothing
                _fill_coupling_from_rk!(h2, d)
            else
                h2.uniform = UniformBlock(h2.row_basis, h2.col_basis,
                                          zeros(Float64, h2.row_basis.k, h2.col_basis.k))
            end
        else
            d = HMatrices.data(hmat)
            if d !== nothing
                h2.dense = Matrix{Float64}(d)
            else
                irange = index_range(h2.row_basis.cluster)
                jrange = index_range(h2.col_basis.cluster)
                h2.dense = zeros(Float64, length(irange), length(jrange))
            end
        end
    elseif !HMatrices.isleaf(hmat) && !isleaf(h2)
        hchildren = HMatrices.children(hmat)
        for i in axes(h2.children, 1)
            for j in axes(h2.children, 2)
                _fill_h2_from_hmat!(h2.children[i, j], hchildren[i, j], row_map, col_map)
            end
        end
    end
end

"""
Compute coupling matrix S = (V_row' A)(V_col' B)' from RkMatrix data.
Uses recursive projection to avoid forming full basis matrices.
"""
function _fill_coupling_from_rk!(h2::H2Matrix, rk)
    rb = h2.row_basis
    cb = h2.col_basis

    if rb.k == 0 || cb.k == 0
        h2.uniform = UniformBlock(rb, cb, zeros(Float64, rb.k, cb.k))
        return
    end

    A = Matrix{Float64}(rk.A)   # |row| × r
    B = Matrix{Float64}(rk.B)   # |col| × r

    # Recursive projection: V_row_full' * A  → (k_row × r)
    VA = _compress_basis_matrix(rb, A)
    # Recursive projection: V_col_full' * B  → (k_col × r)
    VB = _compress_basis_matrix(cb, B)

    # S = VA * VB' = (k_row × r) * (r × k_col) = (k_row × k_col)
    S = VA * VB'

    h2.uniform = UniformBlock(rb, cb, S)
end

"""
    _compress_basis_matrix(cb, M)

Recursively compute `V_full(cb)' * M` without forming the full basis.
`M` must have rows corresponding to `index_range(cb.cluster)`.

Returns a `(k × cols)` matrix.
"""
function _compress_basis_matrix(cb::ClusterBasis, M::AbstractMatrix)
    if isleaf(cb)
        return cb.V' * M
    else
        irange = index_range(cb.cluster)
        result = zeros(Float64, cb.k, size(M, 2))
        for child in cb.children
            cr = index_range(child.cluster)
            local_rows = (cr.start - irange.start + 1):(cr.stop - irange.start + 1)
            child_proj = _compress_basis_matrix(child, view(M, local_rows, :))
            mul!(result, child.E', child_proj, 1.0, 1.0)
        end
        return result
    end
end

# ════════════════════════════════════════════════════════════════════
# Part 2: H²-matrix recompression
# ════════════════════════════════════════════════════════════════════

"""
    recompress!(h2; rtol=1e-8, maxrank=50)

Recompress an H²-matrix in place, reducing basis ranks while maintaining
accuracy within the specified tolerance.

Uses the weight-based recompression algorithm (Börm):
1. Compute basis weights (QR factors encoding basis conditioning)
2. Compute local weights (coupling matrix importance at each cluster)
3. Accumulate total weights (top-down propagation through transfer matrices)
4. Truncate bases (bottom-up SVD using total weights)
5. Project coupling matrices through basis change operators
"""
function recompress!(h2::H2Matrix{N,T};
                     rtol::Float64=1e-8,
                     maxrank::Int=50) where {N,T}
    # ── Step 1: Compute basis weights ──
    col_weights = _compute_basis_weights(h2.col_basis)
    row_weights = _compute_basis_weights(h2.row_basis)

    # ── Step 2: Compute local weights ──
    row_local = _compute_local_weights(h2, col_weights, :row)
    col_local = _compute_local_weights(h2, row_weights, :col)

    # ── Step 3: Accumulate total weights (top-down) ──
    row_total = _accumulate_total_weights(h2.row_basis, row_local)
    col_total = _accumulate_total_weights(h2.col_basis, col_local)

    # ── Step 4: Truncate bases (bottom-up) ──
    row_changes = _truncate_basis!(h2.row_basis, row_total; rtol, maxrank)
    col_changes = _truncate_basis!(h2.col_basis, col_total; rtol, maxrank)

    # ── Step 5: Project coupling matrices ──
    _project_coupling_matrices!(h2, row_changes, col_changes)

    return h2
end

# ─────────────────────────────────────────────────────────────────
# Recompression helpers
# ─────────────────────────────────────────────────────────────────

"""
Compute basis weights (QR R-factors) for each node bottom-up.

At leaf: V = Q*R → weight = R
At non-leaf: QR([R_child1 * E_1; R_child2 * E_2; ...]) → weight = R
"""
function _compute_basis_weights(cb::ClusterBasis{N,T}) where {N,T}
    weights = Dict{UInt,Matrix{Float64}}()
    _compute_weights_recursive!(weights, cb)
    return weights
end

function _compute_weights_recursive!(weights, cb::ClusterBasis)
    if isleaf(cb)
        if cb.k > 0 && size(cb.V, 1) > 0
            F = qr(cb.V)
            k = min(size(F.R, 1), cb.k)
            weights[objectid(cb)] = Matrix(F.R[1:k, 1:cb.k])
        else
            weights[objectid(cb)] = zeros(Float64, cb.k, cb.k)
        end
    else
        for child in cb.children
            _compute_weights_recursive!(weights, child)
        end

        if cb.k > 0
            blocks = Matrix{Float64}[]
            for child in cb.children
                if child.k > 0
                    R_child = weights[objectid(child)]
                    push!(blocks, R_child * child.E)
                end
            end
            if !isempty(blocks)
                M = vcat(blocks...)
                F = qr(M)
                k = min(size(F.R, 1), cb.k)
                weights[objectid(cb)] = Matrix(F.R[1:k, 1:cb.k])
            else
                weights[objectid(cb)] = Matrix{Float64}(I, cb.k, cb.k)
            end
        else
            weights[objectid(cb)] = zeros(Float64, 0, 0)
        end
    end
end

"""
Compute local weights for recompression.

For row basis at cluster t:
  Z_t^+ = QR([ R_{s1} * S_{t,s1}'; R_{s2} * S_{t,s2}'; ... ])
  where R_si is the column basis weight at partner si.

For column basis at cluster s:
  Z_s^+ = QR([ R_{t1} * S_{t1,s}; R_{t2} * S_{t2,s}; ... ])
  where R_ti is the row basis weight at partner ti.
"""
function _compute_local_weights(h2::H2Matrix{N,T},
                                other_weights::Dict{UInt,Matrix{Float64}},
                                side::Symbol) where {N,T}
    local_weights = Dict{UInt,Matrix{Float64}}()
    _collect_local_weights!(local_weights, h2, other_weights, side)

    # QR-compress each accumulated local weight
    for (key, M) in local_weights
        if size(M, 1) > 0 && size(M, 2) > 0
            F = qr(M)
            n_keep = min(size(F.R, 1), size(F.R, 2))
            local_weights[key] = Matrix(F.R[1:n_keep, :])
        end
    end

    return local_weights
end

function _collect_local_weights!(local_weights, h2::H2Matrix{N,T},
                                  other_weights, side) where {N,T}
    if isleaf(h2) && isadmissible(h2) && h2.uniform !== nothing
        rb = h2.row_basis
        cb_col = h2.col_basis
        S = h2.uniform.S

        if side == :row
            # Local weight for row cluster rb:
            # R_col * S' → (k_col × k_row)
            R_col = get(other_weights, objectid(cb_col), nothing)
            if R_col !== nothing && rb.k > 0 && cb_col.k > 0
                contrib = R_col * S'  # (k_col × k_row)
                key = objectid(rb)
                if haskey(local_weights, key)
                    local_weights[key] = vcat(local_weights[key], contrib)
                else
                    local_weights[key] = contrib
                end
            end
        else  # :col
            # Local weight for col cluster cb_col:
            # R_row * S → (k_row × k_col)
            R_row = get(other_weights, objectid(rb), nothing)
            if R_row !== nothing && rb.k > 0 && cb_col.k > 0
                contrib = R_row * S   # (k_row × k_col)
                key = objectid(cb_col)
                if haskey(local_weights, key)
                    local_weights[key] = vcat(local_weights[key], contrib)
                else
                    local_weights[key] = contrib
                end
            end
        end
    elseif !isleaf(h2)
        for child in h2.children
            _collect_local_weights!(local_weights, child, other_weights, side)
        end
    end
end

"""
Accumulate total weights top-down.

Z_root = Z_root^local
Z_child = QR([ Z_child^local; Z_parent * E_child' ])
"""
function _accumulate_total_weights(cb::ClusterBasis{N,T},
                                    local_weights::Dict{UInt,Matrix{Float64}}) where {N,T}
    total_weights = Dict{UInt,Matrix{Float64}}()
    _accumulate_weights_topdown!(total_weights, cb, local_weights, nothing)
    return total_weights
end

function _accumulate_weights_topdown!(total_weights, cb, local_weights, parent_weight)
    local_w = get(local_weights, objectid(cb), nothing)

    if cb.k == 0
        total_weights[objectid(cb)] = zeros(Float64, 0, 0)
    else
        parts = Matrix{Float64}[]

        # Local contribution
        if local_w !== nothing && size(local_w, 2) == cb.k
            push!(parts, local_w)
        end

        # Inherited from parent (propagated through transfer matrix)
        if parent_weight !== nothing && size(parent_weight, 1) > 0 && size(cb.E, 2) > 0
            inherited = parent_weight * cb.E'  # (parent_weight_rows × k_child)
            if size(inherited, 2) == cb.k
                push!(parts, inherited)
            end
        end

        if isempty(parts)
            total_weights[objectid(cb)] = Matrix{Float64}(I, cb.k, cb.k)
        else
            M = vcat(parts...)
            if size(M, 1) > 0
                F = qr(M)
                n_keep = min(size(F.R, 1), size(F.R, 2))
                total_weights[objectid(cb)] = Matrix(F.R[1:n_keep, :])
            else
                total_weights[objectid(cb)] = Matrix{Float64}(I, cb.k, cb.k)
            end
        end
    end

    # Recurse to children
    tw = total_weights[objectid(cb)]
    for child in cb.children
        _accumulate_weights_topdown!(total_weights, child, local_weights, tw)
    end
end

"""
Truncate basis bottom-up using total weights.

Returns a Dict mapping `objectid(cb)` → change operator C (k_new × k_old).

At leaf: M = V * Z', SVD(M), truncate → new V, C = V_new' * V_old
At non-leaf: V̂ = [C_1*E_1; C_2*E_2; ...], M = V̂ * Z', SVD → new E, C
"""
function _truncate_basis!(cb::ClusterBasis{N,T},
                          total_weights::Dict{UInt,Matrix{Float64}};
                          rtol::Float64,
                          maxrank::Int) where {N,T}
    changes = Dict{UInt,Matrix{Float64}}()
    _truncate_recursive!(changes, cb, total_weights; rtol, maxrank)
    return changes
end

function _truncate_recursive!(changes, cb::ClusterBasis{N,T},
                               total_weights; rtol, maxrank) where {N,T}
    if cb.k == 0
        changes[objectid(cb)] = zeros(Float64, 0, 0)
        return
    end

    if isleaf(cb)
        Z = get(total_weights, objectid(cb), nothing)
        V_old = cb.V  # m × k_old

        if Z === nothing || size(Z, 1) == 0
            # No weight info: keep basis as-is
            changes[objectid(cb)] = Matrix{Float64}(I, cb.k, cb.k)
            return
        end

        # M = V_old * Z'
        M = V_old * Z'
        F = svd(M)
        k_new = _truncation_rank(F.S, rtol, maxrank)
        k_new = max(k_new, 0)

        if k_new == 0
            cb.V = zeros(Float64, size(V_old, 1), 0)
            cb.k = 0
            changes[objectid(cb)] = zeros(Float64, 0, size(V_old, 2))
        else
            V_new = F.U[:, 1:k_new]
            # Change operator: maps old coefficients → new coefficients
            C = V_new' * V_old  # (k_new × k_old)
            cb.V = V_new
            cb.k = k_new
            changes[objectid(cb)] = C
        end
    else
        # Process children first (bottom-up)
        for child in cb.children
            _truncate_recursive!(changes, child, total_weights; rtol, maxrank)
        end

        # Form V̂ = [C_1 * E_1; C_2 * E_2; ...]
        k_old = cb.k
        V_hat_blocks = Matrix{Float64}[]
        for child in cb.children
            C_child = changes[objectid(child)]
            if size(C_child, 1) > 0 && size(child.E, 1) > 0
                push!(V_hat_blocks, C_child * child.E)
            else
                push!(V_hat_blocks, zeros(Float64, size(C_child, 1), k_old))
            end
        end

        if isempty(V_hat_blocks) || k_old == 0
            cb.k = 0
            changes[objectid(cb)] = zeros(Float64, 0, k_old)
            for child in cb.children
                child.E = zeros(Float64, child.k, 0)
            end
            return
        end

        V_hat = vcat(V_hat_blocks...)  # (sum_new_child_k × k_old)

        Z = get(total_weights, objectid(cb), nothing)
        if Z === nothing || size(Z, 1) == 0
            # No weight info: keep as-is
            changes[objectid(cb)] = Matrix{Float64}(I, k_old, k_old)
            return
        end

        # M = V̂ * Z'
        M = V_hat * Z'
        F = svd(M)
        k_new = _truncation_rank(F.S, rtol, maxrank)
        k_new = max(k_new, 0)

        if k_new == 0
            cb.k = 0
            changes[objectid(cb)] = zeros(Float64, 0, k_old)
            for child in cb.children
                child.E = zeros(Float64, child.k, 0)
            end
        else
            U_k = F.U[:, 1:k_new]
            # Change operator
            C = U_k' * V_hat  # (k_new × k_old)
            cb.k = k_new
            changes[objectid(cb)] = C

            # Extract new transfer matrices
            offset = 0
            for child in cb.children
                k_child_new = child.k
                child.E = U_k[(offset+1):(offset+k_child_new), :]
                offset += k_child_new
            end
        end
    end
end

"""
Project coupling matrices through basis change operators.

S_new = C_row * S_old * C_col'
"""
function _project_coupling_matrices!(h2::H2Matrix{N,T},
                                      row_changes::Dict{UInt,Matrix{Float64}},
                                      col_changes::Dict{UInt,Matrix{Float64}}) where {N,T}
    if isleaf(h2) && isadmissible(h2) && h2.uniform !== nothing
        rb = h2.row_basis
        cb_col = h2.col_basis
        S_old = h2.uniform.S

        C_row = get(row_changes, objectid(rb), nothing)
        C_col = get(col_changes, objectid(cb_col), nothing)

        if C_row !== nothing && C_col !== nothing &&
           size(C_row, 2) == size(S_old, 1) &&
           size(C_col, 2) == size(S_old, 2)
            S_new = C_row * S_old * C_col'
            h2.uniform = UniformBlock(rb, cb_col, S_new)
        end
    elseif !isleaf(h2)
        for child in h2.children
            _project_coupling_matrices!(child, row_changes, col_changes)
        end
    end
end

# ════════════════════════════════════════════════════════════════════
# Convenience: Adaptive assembly from kernel
# ════════════════════════════════════════════════════════════════════

"""
    assemble_h2matrix_adaptive(K, rowtree, coltree;
        rtol=1e-8, maxrank=50, aca_rtol=nothing, aca_kwargs...)

Assemble an H²-matrix adaptively: first build an H-matrix using ACA,
then convert to H² format with nested bases.

This provides better compression than fixed Chebyshev interpolation
because the ranks adapt to the actual kernel smoothness.

# Arguments
- `K` : kernel matrix
- `rowtree`, `coltree` : cluster trees
- `rtol` : relative tolerance for H² basis truncation
- `maxrank` : maximum rank per cluster
- `aca_rtol` : tolerance for ACA (defaults to `rtol / 10`)
- `aca_kwargs...` : additional arguments for `assemble_hmatrix`
"""
function assemble_h2matrix_adaptive(
    K,
    rowtree::ClusterTree{N,T},
    coltree::ClusterTree{N,T};
    rtol::Float64=1e-8,
    maxrank::Int=50,
    aca_rtol::Union{Float64,Nothing}=nothing,
    adm=StrongAdmissibilityStd(3),
    global_index::Bool=true,
    kwargs...
) where {N,T}
    # Step 1: Build H-matrix via ACA
    ar = aca_rtol === nothing ? rtol / 10 : aca_rtol
    hmat = HMatrices.assemble_hmatrix(K, rowtree, coltree;
                                       adm=adm,
                                       comp=HMatrices.PartialACA(; rtol=ar),
                                       global_index=global_index,
                                       threads=false,
                                       kwargs...)

    # Step 2: Convert to H²
    h2 = compress_hmatrix_to_h2(hmat; rtol, maxrank, _print=false)

    _print_compression_summary(h2)
    return h2
end

"""
    assemble_h2matrix_adaptive(K::AbstractKernelMatrix; kwargs...)

Convenience method that builds cluster trees automatically.
"""
function assemble_h2matrix_adaptive(
    K::AbstractKernelMatrix;
    nmax::Int=32,
    kwargs...
)
    X = map(center, HMatrices.rowelements(K))
    Y = map(center, HMatrices.colelements(K))
    Xclt = ClusterTree(X, GeometricSplitter(; nmax))
    Yclt = ClusterTree(Y, GeometricSplitter(; nmax))
    return assemble_h2matrix_adaptive(K, Xclt, Yclt; kwargs...)
end
