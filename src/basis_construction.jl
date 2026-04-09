"""
    Basis construction algorithms for H²-matrices.

Provides methods to construct the nested cluster bases (leaf bases V and
transfer matrices E) for an H²-matrix from a kernel function. Two approaches
are implemented:

1. **Chebyshev interpolation**: Construct bases using tensor-product Chebyshev
   interpolation on bounding boxes. This is deterministic and well-suited for
   smooth kernels.

2. **Randomized SVD**: Sample the kernel and compute an approximate SVD to
   form the basis. More general but involves randomness.
"""

# ──────────────────────────────────────────────────────────────────
# Chebyshev interpolation basis construction
# ──────────────────────────────────────────────────────────────────

"""
    chebyshev_nodes(n)

Return `n` Chebyshev nodes on [-1, 1].
"""
function chebyshev_nodes(n::Int)
    return [cos((2k - 1) * π / (2n)) for k in 1:n]
end

"""
    chebyshev_nodes_scaled(n, a, b)

Return `n` Chebyshev nodes scaled to interval [a, b].
"""
function chebyshev_nodes_scaled(n::Int, a::Real, b::Real)
    nodes = chebyshev_nodes(n)
    return @. (b - a) / 2 * nodes + (a + b) / 2
end

"""
    chebyshev_interpolation_points(order, bbox::HyperRectangle{N})

Generate tensor-product Chebyshev interpolation points in the given
bounding box. Returns a vector of `SVector{N}` points.
"""
function chebyshev_interpolation_points(order::Int, bbox::HyperRectangle{N,T}) where {N,T}
    lo = low_corner(bbox)
    hi = high_corner(bbox)

    # 1D Chebyshev nodes for each dimension
    nodes_1d = [chebyshev_nodes_scaled(order, lo[d], hi[d]) for d in 1:N]

    # Tensor product
    if N == 1
        return [SVector{1,T}(x) for x in nodes_1d[1]]
    elseif N == 2
        points = SVector{2,T}[]
        for y in nodes_1d[2]
            for x in nodes_1d[1]
                push!(points, SVector{2,T}(x, y))
            end
        end
        return points
    elseif N == 3
        points = SVector{3,T}[]
        for z in nodes_1d[3]
            for y in nodes_1d[2]
                for x in nodes_1d[1]
                    push!(points, SVector{3,T}(x, y, z))
                end
            end
        end
        return points
    else
        error("Chebyshev interpolation only implemented for N ≤ 3")
    end
end

"""
    chebyshev_lagrange_matrix(points, interp_points)

Compute the interpolation matrix L such that for a function f sampled at
`interp_points`, `L * f_interp ≈ f(points)`. This is the Lagrange
interpolation matrix from interpolation points to evaluation points.
"""
function chebyshev_lagrange_matrix(
    points::AbstractVector{<:SVector{N}},
    interp_points::AbstractVector{<:SVector{N}},
    bbox::HyperRectangle{N}
) where {N}
    lo = low_corner(bbox)
    hi = high_corner(bbox)
    npts = length(points)
    ninterp = length(interp_points)

    # Map all points to [-1,1]^N
    function to_reference(pt)
        return SVector{N}(ntuple(d -> 2 * (pt[d] - lo[d]) / (hi[d] - lo[d]) - 1, N))
    end

    # Determine order from the number of interpolation points per dimension
    order = round(Int, ninterp^(1/N))

    # 1D Chebyshev nodes
    cheb_nodes = chebyshev_nodes(order)

    # 1D Lagrange basis values for each point
    L = ones(npts, ninterp)
    for ip in 1:npts
        ref_pt = to_reference(points[ip])
        # Compute tensor product of 1D Lagrange values
        vals_1d = Vector{Vector{Float64}}(undef, N)
        for d in 1:N
            vals_1d[d] = _lagrange_basis_1d(ref_pt[d], cheb_nodes)
        end
        # Tensor product ordering must match interp_points ordering
        if N == 1
            for i1 in 1:order
                L[ip, i1] = vals_1d[1][i1]
            end
        elseif N == 2
            idx = 0
            for i2 in 1:order
                for i1 in 1:order
                    idx += 1
                    L[ip, idx] = vals_1d[1][i1] * vals_1d[2][i2]
                end
            end
        elseif N == 3
            idx = 0
            for i3 in 1:order
                for i2 in 1:order
                    for i1 in 1:order
                        idx += 1
                        L[ip, idx] = vals_1d[1][i1] * vals_1d[2][i2] * vals_1d[3][i3]
                    end
                end
            end
        end
    end
    return L
end

"""
    _lagrange_basis_1d(x, nodes)

Evaluate all Lagrange basis polynomials at point `x` for the given `nodes`.
"""
function _lagrange_basis_1d(x::Real, nodes::Vector{Float64})
    n = length(nodes)
    vals = ones(n)
    for j in 1:n
        for m in 1:n
            if m != j
                vals[j] *= (x - nodes[m]) / (nodes[j] - nodes[m])
            end
        end
    end
    return vals
end

# ──────────────────────────────────────────────────────────────────
# Build cluster bases using Chebyshev interpolation
# ──────────────────────────────────────────────────────────────────

"""
    build_chebyshev_basis!(cb::ClusterBasis, order::Int)

Fill the cluster basis with Chebyshev interpolation bases.
- Leaf nodes get a Lagrange interpolation matrix `V` from the
  interpolation points to the cluster's DOF points.
- Non-leaf (transfer) nodes get a transfer matrix `E` that
  maps from child interpolation to parent interpolation.
"""
function build_chebyshev_basis!(cb::ClusterBasis{N,T}, order::Int) where {N,T}
    k = order^N  # number of interpolation points per cluster

    if isleaf(cb)
        # For leaf: V maps interpolation coefficients → DOF values
        bbox = container(cb.cluster)
        interp_pts = chebyshev_interpolation_points(order, bbox)
        pts = collect(elements(cb.cluster))
        V = chebyshev_lagrange_matrix(pts, interp_pts, bbox)
        set_leaf_basis!(cb, V)
    else
        # First, recurse on children
        for child in cb.children
            build_chebyshev_basis!(child, order)
        end

        # For non-leaf: E maps child interp → parent interp
        # Parent interpolation points
        parent_bbox = container(cb.cluster)
        parent_pts = chebyshev_interpolation_points(order, parent_bbox)

        for child in cb.children
            child_bbox = container(child.cluster)
            child_interp_pts = chebyshev_interpolation_points(order, child_bbox)

            # E: child basis coefficients → parent basis coefficients
            # parent_Lagrange(child_interp_pts) gives us k_child × k_parent
            E = chebyshev_lagrange_matrix(child_interp_pts, parent_pts, parent_bbox)
            set_transfer_matrix!(child, E)
        end

        cb.k = k
    end
    return cb
end

# ──────────────────────────────────────────────────────────────────
# ACA-based basis construction (via column sampling)
# ──────────────────────────────────────────────────────────────────

"""
    build_aca_basis!(cb, K, coltree, adm; rtol, maxrank)

Build cluster basis by sampling the kernel with ACA for each
admissible block and post-processing with SVD truncation.

This constructs an "optimal" nested basis from the low-rank
approximations of each admissible block.
"""
function build_aca_basis!(
    cb::ClusterBasis{N,T},
    K,
    far_blocks::Vector{<:Tuple};
    rtol::Float64=1e-8,
    maxrank::Int=50
) where {N,T}
    # For leaves: aggregate columns from all admissible blocks and compute SVD
    if isleaf(cb)
        _build_leaf_basis_from_blocks!(cb, K, far_blocks; rtol, maxrank)
    else
        # Partition far_blocks to children and recurse
        for child in cb.children
            child_blocks = _filter_blocks_for_cluster(far_blocks, child.cluster)
            build_aca_basis!(child, K, child_blocks; rtol, maxrank)
        end
        # Build transfer matrices from children to parent
        _build_transfer_from_children!(cb; rtol, maxrank)
    end
    return cb
end

function _build_leaf_basis_from_blocks!(
    cb::ClusterBasis{N,T}, K, far_blocks;
    rtol, maxrank
) where {N,T}
    irange = index_range(cb.cluster)
    m = length(irange)

    if isempty(far_blocks)
        # No admissible blocks → rank 0
        cb.V = zeros(Float64, m, 0)
        cb.k = 0
        return cb
    end

    # Collect kernel columns for all far-field partners
    cols = Float64[]
    ncols = 0
    for (_, σ_cluster) in far_blocks
        jrange = index_range(σ_cluster)
        block = Matrix{Float64}(undef, m, length(jrange))
        getblock!(block, K, irange, jrange)
        append!(cols, vec(block))
        ncols += length(jrange)
    end
    C = reshape(cols, m, ncols)

    # Truncated SVD
    F = svd(C)
    # Find truncation rank
    k = _truncation_rank(F.S, rtol, maxrank)
    cb.V = F.U[:, 1:k]
    cb.k = k
    return cb
end

function _build_transfer_from_children!(cb::ClusterBasis; rtol, maxrank)
    # Stack child bases to form the "total" basis at this level
    # then compute transfer matrices via SVD
    child_ranks = [child.k for child in cb.children]
    total_child_k = sum(child_ranks)

    if total_child_k == 0
        cb.k = 0
        for child in cb.children
            child.E = zeros(Float64, child.k, 0)
        end
        return cb
    end

    # Full child basis (block diagonal V_children * I)
    # We need to find a common basis at the parent level
    # Build the "expanded" representation and truncate
    n = length(cb)
    irange = index_range(cb.cluster)

    # Stack full child bases
    V_full = zeros(Float64, n, total_child_k)
    col_offset = 0
    row_offset = 0
    for child in cb.children
        child_n = length(child)
        if child.k > 0
            V_child = isleaf(child) ? child.V : _full_basis(child)
            child_irange = index_range(child.cluster)
            local_rows = (child_irange.start - irange.start + 1):(child_irange.stop - irange.start + 1)
            V_full[local_rows, (col_offset+1):(col_offset+child.k)] = V_child
        end
        col_offset += child.k
    end

    # SVD to find a compressed parent basis
    F = svd(V_full)
    k_parent = _truncation_rank(F.S, rtol, maxrank)
    cb.k = k_parent

    # Transfer matrices: E_child = V_child^T * V_parent
    # Where V_parent = V_full * V * Σ^{-1} (projected)
    # More precisely: V_full ≈ U_k * Σ_k * Vt_k
    # So the parent basis in terms of the original DOFs is U_k
    # Transfer: child's V maps to child DOFs,
    # E_child such that V_child * E_child ≈ U_k restricted to child rows
    U_k = F.U[:, 1:k_parent]

    col_offset = 0
    for child in cb.children
        if child.k > 0
            V_child = isleaf(child) ? child.V : _full_basis(child)
            child_irange = index_range(child.cluster)
            local_rows = (child_irange.start - irange.start + 1):(child_irange.stop - irange.start + 1)
            U_child = U_k[local_rows, :]
            # E = V_child \ U_child (least squares)
            child.E = V_child \ U_child
        else
            child.E = zeros(Float64, 0, k_parent)
        end
        child.k = size(child.E, 1)
        col_offset += child.k
    end

    return cb
end

function _filter_blocks_for_cluster(far_blocks, cluster)
    ir = index_range(cluster)
    return filter(far_blocks) do (τ_cluster, σ_cluster)
        τ_range = index_range(τ_cluster)
        # Check if this cluster is a subset of or equal to τ
        return τ_range.start <= ir.start && ir.stop <= τ_range.stop
    end
end

"""
    _truncation_rank(S, rtol, maxrank)

Determine the truncation rank from a vector of singular values `S`.
"""
function _truncation_rank(S::AbstractVector, rtol::Real, maxrank::Int)
    isempty(S) && return 0
    threshold = rtol * S[1]
    k = 0
    for s in S
        s > threshold || break
        k += 1
    end
    return min(k, maxrank, length(S))
end
