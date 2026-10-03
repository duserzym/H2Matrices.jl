# How accuracy, memory and speed improved

The v0.1.1 numerical changes addressed two separate problems: lost information during nested compression, and unnecessary storage and work when applying the corrected operator. v0.1.2 keeps that implementation and requires Julia 1.13 or later. Raising the Julia requirement is a maintenance decision; the measured performance gains come from the representation and matvec changes described here.

![The progression from correct compression to compact parallel products](assets/advances.svg)

The [PLAG066 validation](validation.md) provides the measured evidence. The [practical guide](accuracy_performance.md) shows how to use these APIs.

## Why a small compression tolerance was insufficient

An H-matrix represents each admissible block independently:

```math
H_{\tau\sigma}=A_{\tau\sigma}B_{\tau\sigma}^{\mathsf T}.
```

H² conversion replaces these independent factors by shared row and column bases, with a coupling for each block:

```math
\widetilde H_{\tau\sigma}=V_\tau S_{\tau\sigma}W_\sigma^{\mathsf T}.
```

A child basis must represent the restrictions of interactions belonging to its ancestors as well as its own interactions. A locally accurate SVD of an incomplete collection of columns cannot preserve a direction that was never supplied to it. This explains why reducing `rtol` alone did not resolve the observed convergence discrepancy.

### Preserve inherited interactions at every level

The earlier internal-basis construction projected a node's direct interaction factors but did not consistently include inherited factors when the node also had direct interactions. The corrected construction combines both collections before projection and SVD.

For a concrete example, suppose an ancestor interaction needs the constant vector on a child cluster, while that child's own interaction needs an alternating-sign vector. Keeping only the child's direct interaction can discard the constant direction even with a tiny local tolerance. The regression test supplies these two directions explicitly and verifies that the child's reconstructed basis retains both.

Nesting is implemented through child transfers:

```math
V_\tau=
\begin{pmatrix}V_{\tau_1}&0\\0&V_{\tau_2}\end{pmatrix}
\begin{pmatrix}E_{\tau_1}\\E_{\tau_2}\end{pmatrix}.
```

Projection uses recursive nested compression rather than repeatedly materializing every full internal basis. That reduces temporary work while retaining the same projection.

### Weight factors by the partner's QR factor

The magnitude of an ACA factor column alone does not measure the importance of its block contribution. Rescaling one rank-one term by

```math
A_{:j}\leftarrow cA_{:j},\qquad B_{:j}\leftarrow B_{:j}/c
```

leaves the block unchanged, but changes an unweighted SVD of the columns of `A`. For row-basis construction, write the partner factor as

```math
B=Q_B R_B,\qquad AB^{\mathsf T}=(AR_B^{\mathsf T})Q_B^{\mathsf T}.
```

The weighted factor `A * R_B'` measures the block through an orthonormal partner. Column construction similarly uses `B * R_A'`. This removes the artificial dependence of the local importance measure on arbitrary ACA factor scaling, up to floating-point effects. A regression rescales ACA factor columns by reciprocal factors of `1e6` and `1e-6`, then checks the represented matrix and conversion accuracy.

### Avoid unused parent bases

A parent with no direct or inherited far-field interactions does not need a basis merely because its children have their own local interactions. The corrected construction gives such a parent rank zero, while retaining the children's active bases and couplings. This avoids large identity embeddings that previously stored directions with no parent-level use.

A zero-rank parent is not an empty subtree. Recompression now traverses its descendants. Otherwise, the corrected zero-rank root could cause an entire active tree to be skipped.

## Make the gradient differentiate the stored energy

Accuracy of the forward product and consistency of the energy derivative are separate requirements. Merrill previously could assemble a separately compressed transposed kernel. That second approximation need not equal the transpose of the stored forward operator.

The package now applies `transpose(h2)` and `adjoint(h2)` using the original bases, transfers, couplings and near-field blocks, reversing the product. For the current real-valued representation, each coupling uses its stored transpose. Rectangular shapes and distinct row/column permutations are respected.

The defining check is

```math
z^{\mathsf T}(\widetilde Hx)
=(\widetilde H^{\mathsf T}z)^{\mathsf T}x.
```

As a simple illustration, the derivative of the stored quadratic energy `E(x) = x' * B * x / 2` is `(B + B') * x / 2`. Replacing `B'` with an independently approximated operator changes that derivative. The actual FEM/BEM energy includes additional maps and solves, whose adjoints must likewise use the stored forward components.

Merrill's integration uses a lazy adjoint of the same H² operator. This also avoids retaining a second independently assembled boundary operator. This consistency improvement is distinct from the 337.78-to-302.75 MB single-operator storage comparison below.

## Make rank caps explicit

`rtol` controls local singular-value truncation; `maxrank` limits the retained rank. If the cap forces removal of a singular value above the requested relative threshold, the implementation reports it. With `strict=true`, assembly/conversion rejects that cap.

Strict recompression performs the operation on a temporary copy and leaves the original operator unchanged if the cap is inadequate. This protects a previously validated operator, at the cost of additional setup memory. Successful recompression requires rebuilding any existing matvec plans.

These checks do not certify a global operator error. ACA error, nested truncations, cancellation, physical weighting and solver sensitivity still require reference comparisons. PLAG066 reached row/column ranks 406/401 in the original tight-tolerance representation; the default cap 120 was not validated for that grain.

## Reuse the work of applying the operator

`H2MatvecPlan` caches permutations, flattens the traversal metadata, and allocates coefficient and physical-vector scratch once. Repeated products reuse that scratch. The upward transform, coupling interactions, downward transform and near-field product still apply the same stored matrix.

This removes the repeated temporary dictionaries and vector allocations of the generic tree-based path. In the recorded PLAG066 run, raw H² products allocated approximately 0.85 MB forward and 1.06 MB adjoint per call. The warmed reusable baseline allocated zero bytes on the measured compiler, before any new compact representation was introduced.

One plan owns mutable scratch. A caller must use `copy(plan)` for each concurrent application; copying shares the numerical matrices and creates independent coefficient, vector and coupling scratch.

## Remove redundant saturated bases algebraically

At very tight tolerances, a basis can have represented rank `k` at least as large as its cluster's physical size `n`. Storing an `n × k` basis provides little compression, even though the basis may be nonorthogonal or rank-deficient.

`H2CompactMatvecPlan` replaces such a basis by an implicit physical-coordinate identity. Let `U` be the original fully expanded basis at a saturated row cluster and `W` the original basis at a saturated column cluster. The block is preserved by

```math
USW^{\mathsf T}=I_n\,(USW^{\mathsf T})\,I_m^{\mathsf T}.
```

If only the row is replaced, its coupling becomes `U * S`; if only the column is replaced, it becomes `S * W'`. A replaced child's transfer to an unreplaced parent becomes `U_child * E_child`. A replaced parent's contribution expands directly into physical output, so its old child transfers are unnecessary. Child-local interactions remain active.

The identity basis is never stored as a dense identity matrix: the upward pass copies physical values and the downward pass adds them directly. No singular values are discarded and no orthogonality assumption is needed. Only floating-point rounding is introduced. Tests cover nonorthogonal and overcomplete interpolation bases as well as adaptive bases.

For the selected PLAG066 operator, 539 row nodes and 537 column nodes use implicit identity bases. Numeric storage falls from 337,784,464 to 302,752,824 bytes at the same ACA/basis tolerances. This saving is a property of the tested operator's saturated bases, not a universal percentage for H² matrices.

## Fold weakly compressing transfers into couplings

Above the saturated levels, a parent basis often compresses its children only mildly. On PLAG036 (12,415 boundary nodes, `eta=3`), depth-4 clusters of about 776 points keep rank about 423 from two children of rank about 249, so their transfer matrices are nearly square, while each such cluster has only a few couplings. These transfers made up about 30% of the compact plan's numbers.

`H2CompactMatvecPlan(h2; passthrough=true)` lets such a node use its children's concatenated coefficients `[x_{c_1}; x_{c_2}]` of dimension `D` as its own coefficients. Its expansion `X = [T_{c_1}; T_{c_2}]` (the children's effective transfers) is folded into its couplings, `X*S` or `S*X'`, and into its own transfer to the parent, `X*E`; the children's transfers are no longer stored. The upward pass copies child coefficients and the downward pass adds them back. A node with rank `k`, parent rank `k_p` and summed coupling-partner width `W` is converted when

```math
(D-k)\,(k_p+W) < D\,k,
```

that is, when this stores fewer numbers. Applying the same rule with `D=|t|` places nearly saturated leaves in physical coordinates. Row decisions are made bottom-up with the column widths of the identity rule, then column decisions with the final row widths; iterating these decisions changed the PLAG066 total by less than 0.1 MB. Like implicit identities, this is exact up to floating-point rounding.

| Operator (`eta=3`, ACA/basis `1e-11/1e-10`) | Compact packet storage | With `passthrough=true` | Forward/adjoint ms (4 workers) |
|---|---:|---:|---|
| PLAG066 (6,028 nodes) | 302.75 MB | 288.08 MB | 3.48/2.73 → 2.86/2.25 |
| PLAG036 (12,415 nodes) | 714.85 MB | 661.02 MB | 8.94/7.38 → 7.47/5.87 |
| PLAG022 (17,875 nodes) | 1511.48 MB | 1400.01 MB | 18.86/15.58 → 14.74/12.38 |

Products changed by at most `1e-15` relative to the original plan, and errors against exact dense products were unchanged. Timings are medians of interleaved runs in one process on a shared machine.

## Pack interactions into larger contiguous products

Many small coupling products have call, indexing and memory-access overhead. `H2PacketMatvecPlan` groups couplings sharing a row coefficient range:

```math
\widehat y_\tau\mathrel{+}=
\begin{bmatrix}S_{\tau\sigma_1}&S_{\tau\sigma_2}&\cdots\end{bmatrix}
\begin{pmatrix}\widehat x_{\sigma_1}\\\widehat x_{\sigma_2}\\\vdots\end{pmatrix}.
```

The input coefficients are gathered into reusable scratch and the contiguous matrix is applied by GEMV. Dense near-field blocks sharing a physical row range are packed similarly. Packing does not truncate matrix entries; it changes evaluation and accumulation order.

The selected operator contains 263 coupling packets and 448 near-field packets. A one-worker packet plan already improves the forward/adjoint medians from 7.79/6.92 ms for the reusable baseline to 6.27/5.50 ms. The next improvement comes from parallel packet execution.

## Parallelize with explicit ownership of writes

Forward coupling packets write disjoint coefficient ranges, so workers can apply different packets directly. Transposed coupling packets can contribute to the same column ranges; each worker therefore accumulates into a private reduction buffer. Reduction occurs in fixed worker order.

Near-field forward packets run in parallel only when their physical output row ranges are disjoint. PLAG066 has overlapping ranges, so this part uses a safe serial fallback. Near-field adjoints use private physical-output buffers and fixed-order reduction. Parallel performance therefore reflects the actual block structure, not an assumption that every phase scales with worker count.

With four workers and one BLAS thread, measured forward/adjoint medians are 3.36/2.61 ms. Small task-scheduling allocations remain: 2,336/4,672 bytes per warmed product in this run. A fixed worker count has a defined reduction order; different counts and BLAS implementations can still change floating-point rounding. The contract is measured accuracy, not bitwise identity with the original traversal.

## Separate representation changes from new approximations

The selected tight-tolerance plan uses `coupling_rtol=nothing`. Reusable scratch, implicit identities and packet packing do not add a truncation tolerance.

`H2LowRankMatvecPlan`, optional compact-plan coupling SVD, relaxed basis tolerances and recompression can reduce storage further, but change the approximation. An earlier 244.40 MB candidate used basis tolerance `1e-7` and had screened torque discrepancy about `3.46e-7 T`. It passed that experiment's `1e-6 T` gate, but does not meet the later `1e-9 T` gate. The 303 MB result retains basis tolerance `1e-10`.

Packet plans keep factorized couplings as factors: left factors join the packet matrix and right factors are applied per segment, so the packet stores the compact plan's numbers (`keep_factors=false` restores re-materialization).

## Truncate and round couplings against the operator scale

A block-relative coupling tolerance resolves weak blocks to a much smaller absolute error than strong ones. With `coupling_scale=:global`, singular values of the compact couplings are discarded below `coupling_rtol` times the largest stored block norm, including the near field. Because many couplings of the saturated levels are physical-coordinate blocks, this acts like a global absolute truncation of those blocks, while the nested bases keep their `rtol`.

`coupling_precision=Float32` additionally stores each retained component whose singular value is below `coupling_rtol*scale/eps(Float32)` in Float32, as factors or as a dense remainder, and keeps the larger components in Float64. `coupling_precision=Float16` adds a third tier: components below `coupling_rtol*scale/eps(Float16)` become Float16 factors with exact power-of-two column scales. The per-component rounding error is then comparable to the discarded components. Products load the stored low-precision numbers and accumulate in Float64, so the adjoint remains the exact transpose of the stored operator up to Float64 rounding; one-worker products remain allocation free.

These are approximations and require the same physical validation as any relaxed tolerance; the measured product errors in the next table were obtained against exact dense products and are not torque errors. Packet plans with four workers, `nmax=32`, ACA `1e-11`:

| Case | Configuration | Storage | Max. relative product error (forward/adjoint) |
|---|---|---:|---|
| PLAG066 | baseline `eta=3`, `rtol=1e-10` | 302.75 MB | 1.85e-11 / 1.72e-11 |
| PLAG066 | `eta=1.5`, `rtol=1e-10`, `passthrough`, global `coupling_rtol=2e-11` | 258.55 MB | 1.62e-11 / 1.65e-11 |
| PLAG066 | `eta=1.5`, `rtol=5e-11`, `passthrough`, global `coupling_rtol=1.5e-11`, Float32 | 190.24 MB | 1.37e-11 / 1.38e-11 |
| PLAG066 | `eta=1.5`, `rtol=5e-11`, `passthrough`, global `coupling_rtol=1e-11`, Float16 | 179.14 MB | 1.22e-11 / 1.26e-11 |
| PLAG036 | baseline `eta=3`, `rtol=1e-10` | 714.85 MB | 7.71e-11 / 7.60e-11 |
| PLAG036 | `eta=1.5`, `rtol=1e-10`, `passthrough`, global `coupling_rtol=2e-11` | 628.53 MB | 6.09e-11 / 6.16e-11 |
| PLAG036 | `eta=1.5`, `rtol=5e-11`, `passthrough`, global `coupling_rtol=1.5e-11`, Float32 | 497.86 MB | 3.35e-11 / 3.27e-11 |
| PLAG036 | `eta=1.5`, `rtol=5e-11`, `passthrough`, global `coupling_rtol=1e-11`, Float16 | 474.95 MB | 3.32e-11 / 3.23e-11 |
| PLAG036 | `eta=1.5`, `rtol=1e-10`, `passthrough`, global `coupling_rtol=3e-11`, Float16 | 438.38 MB | 7.31e-11 / 7.42e-11 |
| PLAG022 | baseline `eta=3`, `rtol=1e-10` | 1511.48 MB | 5.91e-11 / 5.10e-11 |
| PLAG022 | `eta=1.5`, `rtol=5e-11`, `passthrough`, global `coupling_rtol=1.5e-11`, Float32 | 916.00 MB | 2.81e-11 / 2.57e-11 |
| PLAG022 | `eta=1.5`, `rtol=5e-11`, `passthrough`, global `coupling_rtol=1e-11`, Float16 | 853.52 MB | 2.76e-11 / 2.40e-11 |

In the same runs, forward/adjoint packet products of the Float32 variant took 2.84/2.14 ms (PLAG066), 7.64/5.97 ms (PLAG036) and 13.17/10.85 ms (PLAG022), against 3.59/2.86, 8.94/7.38 and 18.86/15.58 ms for the baseline; build time fell by 11-17% because `eta=1.5` makes the basis conversion cheaper. Dense storage is 290.69, 1233.0 and 2556.1 MB for the three grains.

The dense PLAG066 matrix needs 290.69 MB, so only the mixed-precision variants compress it substantially at this accuracy. The physics of the kernel limits compression: at relative accuracy near `1e-11` the far field of clusters below roughly 200 points remains full rank even for `eta=0.5`, so much of the operator is stored as dense or physical-coordinate blocks whatever the admissibility.

## Implementation and background

The numerical changes were released in [v0.1.1](https://github.com/duserzym/H2Matrices.jl/releases/tag/v0.1.1); [v0.1.2](https://github.com/duserzym/H2Matrices.jl/releases/tag/v0.1.2) changes the Julia requirement. Source files describe [nested conversion and recompression](https://github.com/duserzym/H2Matrices.jl/blob/v0.1.2/src/compression.jl), [reusable products](https://github.com/duserzym/H2Matrices.jl/blob/v0.1.2/src/matvec_plan.jl), [compact bases](https://github.com/duserzym/H2Matrices.jl/blob/v0.1.2/src/compact_plan.jl), and [packet execution](https://github.com/duserzym/H2Matrices.jl/blob/v0.1.2/src/packet_plan.jl).

For the broader micromagnetic motivation, see Hertel, Christophersen and Börm, [*Large-scale magnetostatic field calculation in finite element micromagnetics with H2-matrices*](https://arxiv.org/abs/1811.05731). [H2Lib's compression source](https://github.com/H2Lib/H2Lib/blob/master/Library/h2compression.c) is an implementation reference for nested-basis recompression. The accuracy and performance claims on these pages come from this package's tests and PLAG066 measurements, not from assuming the published results transfer to this implementation.
