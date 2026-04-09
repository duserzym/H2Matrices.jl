"""
    H²-Matrix Assembly

Main entry point for constructing an H²-matrix from a kernel function.
The assembly proceeds as:

1. Build cluster trees for row and column points
2. Build cluster bases (nested) from the cluster trees
3. Construct leaf bases using Chebyshev interpolation
4. Build the H²-matrix block structure (admissibility)
5. Fill coupling matrices for admissible blocks (kernel at interpolation points)
6. Fill dense matrices for inadmissible blocks
"""

"""
    assemble_h2matrix(K, rowtree, coltree;
        adm=StrongAdmissibilityStd(3),
        order=4,
        global_index=true)

Assemble an H²-matrix approximation of the kernel matrix `K` using
Chebyshev interpolation of the given `order` for basis construction.

# Arguments
- `K` : kernel matrix (supports `getblock!` or `K[i,j]`)
- `rowtree` : ClusterTree for row indices
- `coltree` : ClusterTree for column indices
- `adm` : admissibility condition (default: strong admissibility with η=3)
- `order` : Chebyshev interpolation order per dimension (default: 4)
- `global_index` : if true, K uses global indexing (default: true)

# Returns
An `H2Matrix` approximating `K`.
"""
function assemble_h2matrix(
    K,
    rowtree::ClusterTree{N,T},
    coltree::ClusterTree{N,T};
    adm=StrongAdmissibilityStd(3),
    order::Int=4,
    global_index::Bool=true
) where {N,T}
    # Wrap K with permuted indexing if global_index
    Kp = if global_index
        HMatrices.PermutedMatrix(K, loc2glob(rowtree), loc2glob(coltree))
    else
        K
    end

    # Step 1: Build cluster bases mirroring the tree structure
    row_basis = build_cluster_basis(rowtree)
    col_basis = build_cluster_basis(coltree)

    # Step 2: Build Chebyshev interpolation bases
    build_chebyshev_basis!(row_basis, order)
    build_chebyshev_basis!(col_basis, order)

    # Step 3: Build block structure
    h2 = build_h2_block_structure(row_basis, col_basis, adm)
    h2.global_index = global_index

    # Step 4: Fill the data (coupling matrices + dense blocks)
    # Extract kernel function and element lists for direct evaluation
    kf = _extract_kernel_info(K)
    _fill_h2_data!(h2, Kp, kf, order)

    _print_compression_summary(h2)
    return h2
end

"""
    _extract_kernel_info(K)

Extract the kernel function and point sets for direct evaluation at
interpolation points. Returns (kernel_func, row_elements, col_elements)
or nothing if not available.
"""
function _extract_kernel_info(K::KernelMatrix)
    return (f=HMatrices.kernel(K), X=HMatrices.rowelements(K), Y=HMatrices.colelements(K))
end
function _extract_kernel_info(K::HMatrices.PermutedMatrix)
    return _extract_kernel_info(K.data)
end
function _extract_kernel_info(K)
    return nothing
end

"""
    assemble_h2matrix(K::AbstractKernelMatrix; order=4, kwargs...)

Convenience method that builds cluster trees automatically from a KernelMatrix.
"""
function assemble_h2matrix(
    K::AbstractKernelMatrix;
    order::Int=4,
    kwargs...
)
    X = map(center, HMatrices.rowelements(K))
    Y = map(center, HMatrices.colelements(K))
    Xclt = ClusterTree(X)
    Yclt = ClusterTree(Y)
    return assemble_h2matrix(K, Xclt, Yclt; order, kwargs...)
end

"""
    _fill_h2_data!(h2, K, kf, order)

Recursively fill all leaf blocks with either coupling matrices (admissible)
or dense matrices (inadmissible).
"""
function _fill_h2_data!(h2::H2Matrix{N,T}, K, kf, order) where {N,T}
    if isleaf(h2)
        if isadmissible(h2)
            _fill_uniform_block!(h2, K, kf, order, N)
        else
            _fill_dense_block!(h2, K)
        end
    else
        for child in h2.children
            _fill_h2_data!(child, K, kf, order)
        end
    end
    return h2
end

"""
    _fill_uniform_block!(h2, K, kf, order, dim)

For an admissible block (τ,σ), compute the coupling matrix S such that:
    K|_{τ×σ} ≈ V_τ * S * V_σ'

For Chebyshev interpolation, S[α,β] = kernel(ξ_α^τ, ξ_β^σ) where ξ^τ,ξ^σ
are the Chebyshev interpolation points in the respective bounding boxes.
"""
function _fill_uniform_block!(h2::H2Matrix{N,T}, K, kf, order, dim) where {N,T}
    rb = h2.row_basis
    cb_col = h2.col_basis
    k_row = rb.k
    k_col = cb_col.k

    if k_row == 0 || k_col == 0
        h2.uniform = UniformBlock(rb, cb_col, zeros(Float64, k_row, k_col))
        return h2
    end

    row_bbox = container(rb.cluster)
    col_bbox = container(cb_col.cluster)

    order_row = _order_from_rank(k_row, dim)
    order_col = _order_from_rank(k_col, dim)

    row_interp_pts = chebyshev_interpolation_points(order_row, row_bbox)
    col_interp_pts = chebyshev_interpolation_points(order_col, col_bbox)

    # Evaluate kernel directly at interpolation points
    S = Matrix{Float64}(undef, k_row, k_col)
    if kf !== nothing
        # Direct kernel evaluation
        for j in 1:k_col
            for i in 1:k_row
                S[i, j] = kf.f(row_interp_pts[i], col_interp_pts[j])
            end
        end
    else
        # Fallback: use nearest DOF points as proxy
        for j in 1:k_col
            for i in 1:k_row
                ri = _find_nearest_index(row_interp_pts[i], rb.cluster)
                ci = _find_nearest_index(col_interp_pts[j], cb_col.cluster)
                S[i, j] = K[ri, ci]
            end
        end
    end

    h2.uniform = UniformBlock(rb, cb_col, S)
    return h2
end

"""
    _fill_dense_block!(h2, K)

For an inadmissible block (τ,σ), store the dense matrix K[τ,σ].
"""
function _fill_dense_block!(h2::H2Matrix{N,T}, K) where {N,T}
    irange = index_range(h2.row_basis.cluster)
    jrange = index_range(h2.col_basis.cluster)
    m = length(irange)
    n = length(jrange)
    D = Matrix{Float64}(undef, m, n)
    getblock!(D, K, irange, jrange)
    h2.dense = D
    return h2
end

"""
Determine the Chebyshev order from rank k and dimension N.
Since k = order^N, order = round(Int, k^(1/N)).
"""
function _order_from_rank(k::Int, N::Int)
    order = round(Int, k^(1/N))
    while order^N < k
        order += 1
    end
    return order
end

"""
Find the index of the nearest point in the cluster to a given target point.
"""
function _find_nearest_index(target::SVector{N}, cluster::ClusterTree{N}) where {N}
    els = elements(cluster)
    irange = index_range(cluster)
    best_dist = Inf
    best_idx = first(irange)
    for (i, el) in zip(irange, els)
        d = norm(el - target)
        if d < best_dist
            best_dist = d
            best_idx = i
        end
    end
    return best_idx
end
