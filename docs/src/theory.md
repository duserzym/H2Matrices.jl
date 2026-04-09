# Theory

This page walks through the mathematical ideas behind H²-matrices, starting
from the basics. No prior knowledge of hierarchical matrices is assumed — just
familiarity with linear algebra and the idea of a kernel function.

## The Problem: Dense Kernel Matrices

Suppose you have ``N`` source points ``\{y_j\}`` and ``N`` target points
``\{x_i\}``, and a kernel function ``G(x, y)`` (e.g., the Laplace Green's
function ``G(x,y) = 1/(4\pi\|x-y\|)``).  The kernel matrix

```math
K_{ij} = G(x_i, y_j), \qquad i,j = 1,\ldots,N
```

is ``N \times N`` and dense.  Storing it costs ``O(N^2)`` memory, and
multiplying it by a vector costs ``O(N^2)`` operations.  For large ``N``
(e.g., ``10^5`` to ``10^7``), this is prohibitive.

## Low-Rank Approximation of Far-Field Blocks

The key observation is that ``G(x, y)`` is **smooth** when ``x`` and ``y`` are
far apart.  If we partition our points into clusters and pick two clusters
``\tau`` (rows) and ``\sigma`` (columns) that are well-separated, the submatrix
``K|_{\tau \times \sigma}`` can be approximated by a low-rank factorization:

```math
K|_{\tau \times \sigma} \approx A \, B^\top, \qquad A \in \mathbb{R}^{|\tau| \times k}, \quad B \in \mathbb{R}^{|\sigma| \times k}
```

where the rank ``k`` is small and independent of ``N``.  This is the foundation
of all hierarchical matrix methods (H-matrices, FMM, etc.).

### Admissibility

Two clusters ``\tau`` and ``\sigma`` are **admissible** (eligible for low-rank
approximation) if they satisfy a separation condition.  The standard criterion
is:

```math
\min(\mathrm{diam}(\tau),\, \mathrm{diam}(\sigma)) \leq \eta \cdot \mathrm{dist}(\tau, \sigma)
```

where ``\eta > 0`` is a parameter (typically ``\eta = 2`` or ``3``).  Blocks that
are not admissible (near-field blocks) are stored as dense matrices.

## Cluster Trees and Block Trees

A **cluster tree** is built by recursively bisecting the point set
(e.g., along the longest bounding-box dimension) until each leaf cluster
contains at most ``n_{\max}`` points.

The **block tree** is then formed by testing admissibility at each level.  For a
pair of clusters ``(\tau, \sigma)``:
- If admissible → store as low-rank (far-field block).
- If not admissible and both are leaves → store as dense (near-field block).
- Otherwise → recurse into children.

This gives a hierarchical partition of the full matrix into ``O(N)`` blocks.

## From H-Matrices to H²-Matrices

In a standard **H-matrix**, each admissible block stores its own independent
low-rank factors ``A_b, B_b``.  This gives ``O(N \log N)`` storage and
matvec cost in general.

An **H²-matrix** improves on this by introducing **shared nested bases**.
Instead of storing separate factors per block, all blocks at the same level
share a common **cluster basis**:

```math
K|_{\tau \times \sigma} \approx V_\tau \, S_{\tau\sigma} \, W_\sigma^\top
```

where:
- ``V_\tau \in \mathbb{R}^{|\tau| \times k_\tau}`` is the **row cluster basis**
  (shared by all blocks with row cluster ``\tau``).
- ``W_\sigma \in \mathbb{R}^{|\sigma| \times k_\sigma}`` is the **column cluster
  basis** (shared by all blocks with column cluster ``\sigma``).
- ``S_{\tau\sigma} \in \mathbb{R}^{k_\tau \times k_\sigma}`` is the small
  **coupling matrix** (unique to each block).

### Nested (Hierarchical) Bases

The bases are **nested**: a parent cluster's basis is built from its children's
bases via small **transfer matrices**.  For a cluster ``\tau`` with children
``\tau_1, \tau_2``:

```math
V_\tau = \begin{pmatrix} V_{\tau_1} \\ V_{\tau_2} \end{pmatrix}
\begin{pmatrix} E_{\tau_1} & 0 \\ 0 & E_{\tau_2} \end{pmatrix}
```

where ``E_{\tau_i} \in \mathbb{R}^{k_{\tau_i} \times k_\tau}`` are the transfer
matrices.  This nesting means:
- Only **leaf** clusters store their full basis matrix ``V`` (size ``|t| \times k``).
- Non-leaf clusters store only the small transfer matrices ``E`` (size ``k_{\text{child}} \times k_{\text{parent}}``).
- The total storage for all bases is ``O(N)``.

## Matrix–Vector Product: O(N)

The H²-matrix–vector product ``y = K x`` uses three phases:

### Phase 1: Forward (Upward) Transform

Project the input vector onto the column basis, bottom-up:

```math
\hat{x}_\sigma = \begin{cases}
  W_\sigma^\top x|_\sigma & \text{if } \sigma \text{ is a leaf} \\
  \sum_i E_{\sigma_i}^\top \hat{x}_{\sigma_i} & \text{otherwise (transfer from children)}
\end{cases}
```

### Phase 2: Coupling Interaction

For each admissible block, apply the small coupling matrix:

```math
\hat{y}_\tau \mathrel{+}= S_{\tau\sigma} \, \hat{x}_\sigma
```

### Phase 3: Backward (Downward) Transform

Expand the result back to physical space, top-down:

```math
y|_\tau \mathrel{+}= \begin{cases}
  V_\tau \hat{y}_\tau & \text{if } \tau \text{ is a leaf} \\
  \text{propagate: } \hat{y}_{\tau_i} \mathrel{+}= E_{\tau_i} \hat{y}_\tau & \text{then recurse}
\end{cases}
```

The dense (near-field) blocks are handled by direct matrix–vector multiplication.

Each phase visits ``O(N)`` data, giving an overall **O(N)** matrix–vector product.

## Basis Construction Strategies

H2Matrices.jl provides two approaches:

### 1. Chebyshev Interpolation

For a smooth kernel, we can approximate it by polynomial interpolation.
On each cluster's bounding box, we place tensor-product **Chebyshev nodes**
``\{\xi_\alpha\}`` and express:

```math
G(x, y) \approx \sum_\alpha \sum_\beta L_\alpha(x) \, G(\xi_\alpha, \xi_\beta) \, L_\beta(y)
```

where ``L_\alpha`` are Lagrange interpolation polynomials.  This gives:
- **Leaf basis**: ``V_\tau = [L_\alpha(x_i)]_{i,\alpha}`` (Lagrange matrix).
- **Coupling matrix**: ``S_{\tau\sigma} = [G(\xi_\alpha^\tau, \xi_\beta^\sigma)]_{\alpha,\beta}``
  (kernel evaluated at interpolation points).
- **Transfer matrix**: Lagrange interpolation from child nodes to parent nodes.

The rank ``k = p^d`` where ``p`` is the interpolation order and ``d`` is the
spatial dimension.

### 2. Adaptive (ACA → SVD)

For more general or less smooth kernels, we can build the basis adaptively:

1. Assemble an H-matrix using **Adaptive Cross Approximation** (ACA), which
   finds per-block low-rank factors ``A_b B_b^\top`` with adaptive ranks.
2. Collect all the ``A`` columns for each row cluster and compute an SVD to find
   a shared basis that captures the column space of all blocks at that cluster.
3. Build transfer matrices by projecting through children's bases.
4. Compute coupling matrices by projecting the original low-rank data through
   the nested bases.

This often achieves better compression than fixed-order Chebyshev interpolation
because the ranks adapt to the actual smoothness of the kernel.

## Recompression

An H²-matrix may have higher ranks than necessary (e.g., from conservative
Chebyshev orders or ACA tolerances).  **Recompression** reduces the ranks
while controlling the error:

1. **Basis weights**: Compute QR factors that encode the conditioning of each
   basis.
2. **Local weights**: Measure the importance of each cluster's basis based on
   the coupling matrices it appears in.
3. **Total weights**: Propagate importance top-down through transfer matrices.
4. **Truncation**: Perform weighted SVD at each cluster (bottom-up) and truncate
   to the desired tolerance.
5. **Coupling update**: Project all coupling matrices through the basis change
   operators.

This is the algorithm from Börm's *Efficient Numerical Methods for Non-local
Operators*.

## Complexity Summary

| Operation          | H-matrix             | H²-matrix     |
|:-------------------|:---------------------|:---------------|
| Storage            | ``O(N k \log N)``    | ``O(N k)``     |
| Matrix–vector      | ``O(N k \log N)``    | ``O(N k)``     |
| Assembly (Cheb.)   | —                    | ``O(N k)``     |
| Assembly (ACA→H²)  | ``O(N k \log N)``    | + ``O(N k)``   |

where ``k`` is the typical block rank.

## References

- S. Börm, *Efficient Numerical Methods for Non-local Operators: H²-Matrix
  Compression, Algorithms and Analysis*, EMS, 2010.
- S. Börm, L. Grasedyck, W. Hackbusch, "Hierarchical Matrices",
  *Lecture Notes*, Max Planck Institute, 2003.
- W. Hackbusch, *Hierarchical Matrices: Algorithms and Analysis*, Springer, 2015.
