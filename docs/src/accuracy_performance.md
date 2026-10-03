# Accuracy and performance

Use separate controls for ACA accuracy, nested-basis truncation and optional coupling truncation. For repeated products, choose a reusable representation and validate both its forward product and the adjoint used by the application. [How the advances work](advances.md) explains the algebra and implementation; [PLAG066 validation](validation.md) gives the measured results and their scope.

## Assemble with explicit error controls

Julia 1.13 or later is required for v0.1.2. `K` below is an existing `HMatrices.KernelMatrix`:

```julia
using H2Matrices, LinearAlgebra
h2 = assemble_h2matrix_adaptive(K;
    rtol=1e-10, aca_rtol=1e-11, maxrank=512,
    nmax=32, strict=true)
```

| Control | What it controls | What it does not establish |
|---|---|---|
| `aca_rtol` | Independent H-block ACA construction | Total physical-field or solver error |
| `rtol` | Local nested-basis singular-value truncation | A global relative error bound |
| `maxrank` | Maximum retained basis rank | Accuracy if the required rank exceeds the cap |
| `strict=true` | Reject a cap that violates a requested local threshold | Mesh, path or whole-campaign convergence |
| `coupling_rtol` | Additional local coupling SVD truncation when requested | The original stored operator's accuracy without new checks |
| `coupling_precision=Float32` or `Float16` | Float32/Float16 storage (range-scaled) of small retained coupling components, Float64 arithmetic (requires `coupling_rtol`) | Accuracy without new checks; the rounding error is estimated, not bounded |
| `passthrough=true` | Exact folding of weakly compressing transfers into couplings (compact/packet plans) | Nothing beyond rounding without `coupling_rtol`; with it, the folded couplings are truncated differently |
| `nmax`, admissibility | Tree/block partitioning and resulting ranks/work | A mesh-independent best configuration |
| `threads=true` | Builds the intermediate H-matrix leaves on all Julia threads; the result equals the serial build (`K` must allow concurrent `getblock!`) | Any accuracy change |
| `comp` | Replaces the default `PartialACA(; rtol=aca_rtol)` H-block compressor | Accuracy of the substitute compressor |
| `conversion_threads` | Runs the H → H² conversion as Julia tasks (default with several Julia threads and one BLAS thread); bitwise independent of the thread count | Any accuracy change |
| `error_control=:global` | Measures ACA and basis truncations against one operator scale `s` (absolute tolerances `aca_rtol*s` and `rtol*s`) instead of each block's or cluster's own norm | That weak blocks need less relative accuracy in a given application |
| `atol`, `aca_atol`, `safeguard_rtol` | Explicit absolute basis and ACA thresholds, and a relative floor for the basis threshold | A global error bound |
| `coupling_scale` | Reference of the optional coupling truncation (requires `coupling_rtol`): the coupling's own norm (`:block`), the largest stored block norm (`:global`) or a given scale `s` (threshold `coupling_rtol*s`) | A global error bound; the original stored operator's accuracy without new checks |
| `precision_rtol`, `format48` | Float32 (and 48-bit) storage of low-weight rotated packet rows in `H2MixedPacketMatvecPlan`, under the a priori bound `‖Ã - A‖_F ≤ precision_rtol · ‖A‖_F` relative to the Float64 compact operator | A per-vector or physical error bound; the compact operator's own accuracy |

Kernel evaluation usually dominates the H-matrix build. A kernel type can
specialize `HMatrices.getblock!` for `HMatrices.PermutedMatrix{<:MyKernel}`
columns, for `adjoint` rows and for `UnitRange × UnitRange` dense blocks.
`assemble_hmatrix(...; global_index=true)` reaches the kernel only through
those calls, and a specialized kernel can share work between entries.

These are the validated PLAG066 parameters, not universal defaults. Tightening `rtol` cannot repair an insufficient rank cap, incomplete basis construction, or an energy-gradient inconsistency. Near-field blocks remain dense.

## Choose the product representation

| Representation | Main purpose | Additional approximation | Original H² object retained? |
|---|---|---|---|
| `H2Matrix` | Assembly, diagnostics, recompression and products | ACA/basis construction already chosen | It is the original object |
| `H2MatvecPlan(h2)` | Cache traversal/permutations and reuse scratch | None | Yes |
| `H2CompactMatvecPlan(h2)` | Also remove saturated bases via implicit identities | None by default | No object reference; some matrices are shared |
| `H2PacketMatvecPlan(h2; workers=1)` | Compact bases plus contiguous interaction packets | None by default | No object reference; some matrices are shared |
| `H2LowRankMatvecPlan(h2; rtol=...)` | Factor selected couplings to reduce storage | Local SVD truncation | No object reference; some matrices are shared |
| `H2MixedPacketMatvecPlan(h2; workers=1, precision_rtol=1e-13)` | Packet plan with low-weight coupling rows in Float32 (optionally 48-bit) | Bounded reduced-precision rounding (`precision_rtol`) | No object reference; some matrices are shared |

Compact, packet and low-rank plans are matvec-only operators; they do not provide general matrix indexing or recompression. Keep an assembly representation only if those operations are needed. Plans retain the matrices and metadata needed for their products, so they remain usable after the source object goes out of scope.

## Repeated products and stored adjoints

```julia
BLAS.set_num_threads(1)
plan = H2PacketMatvecPlan(h2; workers=4)
x = randn(size(plan, 2))
z = randn(size(plan, 1))
y = zeros(size(plan, 1))
g = zeros(size(plan, 2))

mul!(y, plan, x)
mul!(g, adjoint(plan), z)

# Several right-hand sides: the operator is streamed once per block of up to 16 vectors.
X = randn(size(plan, 2), 9)
Y = zeros(size(plan, 1), 9)
mul!(Y, plan, X)
mul!(X, adjoint(plan), Y)
```

Multi-vector products use a workspace that is allocated on first use and kept by the plan. `multi_workspace_bytes(plan, k)` reports its size for `k` vectors and `multi_workspace_bytes(plan)` the size currently kept; it grows with `k` up to the 16-vector block width. `release_multi_workspace!(plan)` frees it (the next multi-vector product allocates one for its own width again).

Use `adjoint(plan)` for the adjoint of this same stored approximation. Do not independently compress a transposed kernel and assume it is identical. A useful verification is `dot(z, plan*x) ≈ dot(adjoint(plan)*z, x)`, with a tolerance appropriate for accumulated rounding.

Use one BLAS thread when evaluating packet-worker speedups. Multiple packet workers and multiple BLAS threads can oversubscribe the processor. `workers=1` is the default; measure worker counts on the target grain and hardware. A one-worker warmed plan can avoid per-call allocations, while threaded paths create small scheduling objects. Packet products are bitwise independent of the worker count, so the worker count can be tuned without changing results.

## Retain only the operator you need

A standalone packet plan's storage reduction does not reduce the total live memory if the caller also retains H, H² and intermediate plans. For a product-only workload, limit their lifetimes:

```julia
function build_operator(K)
    h2 = assemble_h2matrix_adaptive(K;
        rtol=1e-10, aca_rtol=1e-11,
        maxrank=512, nmax=32, strict=true)
    return H2PacketMatvecPlan(h2; workers=4, consume=true)
end
plan = build_operator(K)
```

`assemble_h2matrix_adaptive` releases its private intermediate H-matrix block by block during the conversion, and `consume=true` releases the raw H² blocks while they are packed, so the H-matrix, raw H² operator and packets are never all alive together. Both are exact (the plan is bitwise identical). Construction still needs the H-matrix itself, basis/SVD workspaces and one packet at a time; reclamation is governed by Julia's garbage collector, which the consuming builders prompt with a few cheap full collections. Use `consume=true` only when the source operator is no longer needed.

The conversion condenses ancestor interactions exactly (Gram-preserving QR of each cluster's active set), stores untruncated bases as identities, and runs independent subtrees and couplings as Julia tasks when Julia has several threads and BLAS uses one (`compress_hmatrix_to_h2(...; threads=true)`, `conversion_threads=true` in `assemble_h2matrix_adaptive`; results are bitwise independent of the thread count). Truncation ranks match the v0.1.3 construction except for rounding-level decisions at the threshold.

`storage_bytes(plan)` counts numerical basis, transfer, coupling and near-field data, excluding scratch and metadata. `Base.summarysize(plan)` estimates retained Julia object size, including scratch/metadata; it is not process RSS and does not include unrelated FEM data or BLAS workspaces. Summing `summarysize` across copies can double-count shared matrices.

## Independent concurrent callers

A plan contains mutable scratch and cannot serve concurrent products by itself. Use independent copies:

```julia
callers = [copy(plan) for _ in 1:2]
inputs = [randn(size(plan, 2)) for _ in callers]
outputs = [zeros(size(plan, 1)) for _ in callers]
@sync for i in eachindex(callers)
    Threads.@spawn mul!(outputs[i], callers[i], inputs[i])
end
```

Copies share numerical matrices and traversal metadata, but own coefficient/vector scratch, factorized-coupling scratch, packet slot buffers and the multi-vector workspace. Do not mutate shared data while any caller is running. When outer tasks already use the available CPU budget, one packet worker per caller can avoid excessive nested parallelism.

Rebuild plans after successful recompression or source changes. `copy(plan)` creates a new workspace for the same representation; it does not rebuild it against a changed source.

## Optional approximation reductions

```julia
compact = H2CompactMatvecPlan(h2; coupling_rtol=1e-10)
# Alternatively:
factored = H2LowRankMatvecPlan(h2; rtol=1e-10)
```

These options perform local coupling SVD truncation and require another accuracy check.

For product-only use, the exact `passthrough=true` option and the approximate global and mixed-precision coupling options compose:

```julia
compact = H2CompactMatvecPlan(h2; passthrough=true,
    coupling_rtol=1e-11, coupling_scale=:global, coupling_precision=Float16)
plan = H2PacketMatvecPlan(compact; workers=4)
```

[How the advances work](advances.md) lists measured storage and product errors for these settings. Recompression and relaxed basis tolerances likewise need a physical error budget. Do not infer equivalent torque accuracy from equivalent storage size or from the same numerical tolerance at different stages.

Strict recompression is transactional:

```julia
recompress!(h2; rtol=1e-9, maxrank=512, strict=true)
# Rebuild any plans after success.
```

A rejected strict rank cap leaves `h2` unchanged. The temporary copy increases setup memory. Conversion and recompression traverse active children even when their parent has rank zero.

### Global (absolute) error control

By default every truncation is relative to what it acts on: ACA stops when the
last update is small relative to the block's estimated norm, a cluster basis
discards singular values below `rtol` times its own largest one, and
`coupling_rtol` is relative to each coupling's norm. Global control measures
them against one operator scale instead, so a block whose norm is far below
the operator's is not resolved to the same relative accuracy as the strongest
blocks:

```julia
s = estimate_operator_scale(K, rowtree, coltree)  # RMS row norm from 32 sampled rows
h2 = assemble_h2matrix_adaptive(K, rowtree, coltree;
    error_control=:global, scale=s,                # ACA atol = aca_rtol*s, basis atol = rtol*s
    rtol=5e-11, aca_rtol=1e-11, maxrank=2048, strict=true,
    adm=HMatrices.StrongAdmissibilityStd(1.5))
compact = H2CompactMatvecPlan(h2; coupling_rtol=3e-11, coupling_scale=s)
plan = H2PacketMatvecPlan(compact; workers=4)     # keeps the factored couplings
```

`compress_hmatrix_to_h2(H; rtol, atol, safeguard_rtol)` exposes the basis rule
directly: keep singular values above `max(rtol*σ₁, min(atol, safeguard_rtol*σ₁))`.
`atol=0` is the default block-relative rule, `rtol=0` is purely absolute, and a
finite `safeguard_rtol` keeps every cluster resolved to at least that relative
level if the scale is too large. These options change the approximation and
need the same validation as any tolerance change.

Global control pays off when block norms span many orders of magnitude. For the
double-layer boundary operator of the micromagnetic campaign it does not: the
Frobenius norm of an admissible block of a 1/r² kernel on a surface is roughly
independent of the block's level and size, so admissible block norms stay
within about two decades (0.005-0.6 on PLAG066, against an RMS row norm of
0.62 on PLAG066 and PLAG036), and block-relative and absolute thresholds select
nearly the same ranks. Measured at equal accuracy (forward and adjoint product
errors against exact dense products no larger than the validated baseline's):

- ACA: an absolute tolerance shrank the intermediate H-matrix by up to about
  4.5% (PLAG066 378 → 361 MB, PLAG022 2027 → 1948 MB); the final operator's
  storage did not change.
- Bases: absolute truncation selected nearly the same ranks as relative
  truncation (PLAG066: 30921 against 31030 basis vectors, with a larger
  error). On PLAG036 it matched relative truncation at the same error and
  needed 2-3% more storage at relaxed accuracy.
- Couplings: truncation itself is the useful step. With `eta=1.5`, one
  setting per variant on all grains (`rtol=1e-10`, `aca_rtol=1e-11`, packet
  plans):

  | Coupling truncation | PLAG066 | PLAG036 | PLAG022 |
  |---|---|---|---|
  | none, `eta=3` (validated baseline) | 302.8 MB, 1.85e-11 | 714.8 MB, 7.7e-11 | 1511 MB, 5.9e-11 |
  | `coupling_rtol=1e-10` (`:block`) | 274.8 MB, 1.36e-11 | 668.9 MB, 6.2e-11 | 1309 MB, 5.6e-11 |
  | `coupling_rtol=2e-11, coupling_scale=:global` | 272.3 MB, 1.63e-11 | 668.6 MB, 6.2e-11 | 1303 MB, 5.5e-11 |

  (largest of the forward and adjoint relative errors). A global scale saved
  0-1% over block-relative truncation at equal accuracy, and 1-2.5% at relaxed
  accuracy (1e-10 to 1e-8) on PLAG066.

What global control does change is how the error scales with the grain. With
block-relative tolerances (`rtol=1e-10`, `coupling_rtol=1e-10`) the product
error grew from 1.4e-11 (PLAG066) to 6.2e-11 (PLAG036) and 5.6e-11
(PLAG022), consistent with each cluster's reference norm σ₁ growing with the
number of blocks in its (inherited) block row. With
`error_control=:global` (`rtol=5e-11`, `aca_rtol=1e-11`) and
`coupling_rtol=3e-11, coupling_scale=s` it stayed at 1.5e-11, 1.9e-11 and
2.4e-11 on PLAG066, PLAG036 and PLAG022 (271.9, 703.3 and 1345 MB), so one
setting holds an accuracy target across grain sizes; at that tighter accuracy
block-relative control needed the same storage (PLAG036: 703.7 MB at 1.8e-11).
## Mixed-precision packet storage

```julia
BLAS.set_num_threads(1)
plan = H2MixedPacketMatvecPlan(h2; workers=4, precision_rtol=1e-13, passthrough=true, consume=true)
precision_summary(plan)        # bound, reference norm, Float32 rows, byte counts
```

Each coupling packet is rotated to its left singular basis; rows of small singular weight are stored in Float32 (with `format48=true` also in a 48-bit format) and every product accumulates in Float64. The rows are chosen to minimize bytes under the a priori bound `‖Ã - A‖_F ≤ precision_rtol · η` on the reduced-precision rounding, where `η` is the Frobenius norm of the stored blocks (`‖A‖_F` for orthonormal bases). It is a root-mean-square bound on product perturbations, not a per-vector guarantee, and it does not include the Float64 roundoff of the rotations themselves. The near field stays in Float64.

The plan runs on the packet-plan engine with the same kernels: forward and adjoint products apply the same stored values (the adjoint is an exact transpose), products are bitwise independent of the worker count, several right-hand sides are supported, and `precision_rtol=0` reproduces `H2PacketMatvecPlan` bitwise. Construction is deterministic under threads. From an H² matrix the plan accepts the compact options `passthrough`, `coupling_rtol` and `coupling_scale` (with `consume=true` the source blocks are released while the plan is built). Pass-through composes exactly and is recommended. Factorized couplings from `coupling_rtol` are multiplied out in the packets, so they bring no storage saving here; `coupling_precision` is a different reduced-precision storage and is rejected.

Measured on the PLAG066/PLAG036 boundary operators (nb 6028/12415; `nmax=32`, `aca_rtol=1e-11`, `rtol=1e-10`, strict ranks; largest relative forward/adjoint errors against exact dense products over four reference vectors -- three Gaussian and one smooth. Smooth inputs and single columns see errors 1.5-4.5x larger, so read these as comparative levels, not bounds):

| Configuration | Operator MB (PLAG066 / PLAG036) | Error, PLAG066 | Error, PLAG036 |
|---|---|---|---|
| packet plan, `eta=3` (Float64) | 302.8 / 714.8 | 1.850e-11 / 1.717e-11 | 7.705e-11 / 7.601e-11 |
| `1e-13` | 256.8 / 637.2 | 1.850e-11 / 1.717e-11 | 7.705e-11 / 7.601e-11 |
| `1e-13`, `format48=true` | 225.2 / 578.1 | 1.850e-11 / 1.717e-11 | 7.705e-11 / 7.601e-11 |
| `1e-13`, `passthrough=true` | 241.7 / 582.1 | 1.850e-11 / 1.717e-11 | 7.705e-11 / 7.601e-11 |
| `1e-13`, `format48=true`, `passthrough=true` | 209.8 / 521.3 | 1.850e-11 / 1.717e-11 | 7.705e-11 / 7.601e-11 |
| `eta=1.5`, `1e-13`, `passthrough=true` | 245.9 / 578.5 | 1.155e-11 / 9.68e-12 | 5.902e-11 / 6.008e-11 |
| `eta=1.5`, `1e-13`, `format48=true`, `passthrough=true` | 211.4 / 517.2 | 1.155e-11 / 9.68e-12 | 5.902e-11 / 6.008e-11 |

The difference from the Float64 operator is about `4e-14` relative, three orders of magnitude below the compression error, so errors against the dense operator change by at most `5e-5` of their value (in either direction; with `eta=3` this can exceed the validated errors in the last digits). With `eta=1.5` the underlying operator is more accurate than the validated one, and the bounded rounding keeps it so.

Single-vector products read fewer bytes and are memory-bound, so they get faster: with four workers and pass-through, 1.35/1.27 ms forward/adjoint against 1.64/1.44 ms for the validated packet plan on PLAG066 and 3.96/3.54 against 4.46/4.06 ms on PLAG036. With several right-hand sides the kernels are compute-bound; the mixed plan does the same number of multiply-adds plus the widening, so per-vector times with nine vectors were up to 16% slower than the packet plan's. `format48=true` saves another 10-14% of operator bytes, with single-vector products from 3% slower to 11% faster than the packet plan's and 27-54% slower multi-vector products. Looser `precision_rtol` saves more bytes (`1e-12`: about 7% more without pass-through) but starts to change the operator at the compression-error level; validate it before use. [Compare the variants](advances.md#Compare-the-variants) lists all timings.

## Choose a configuration

All of the following use one BLAS thread, `nmax=32`, `aca_rtol=1e-11`, `strict=true` and a rank cap that is not reached, and meet the validated product-error levels on PLAG066 and PLAG036 as measured by the four-reference-vector maximum and the Frobenius (RMS) error (see the tables above and in [How the advances work](advances.md#Compare-the-variants)). That is not a worst-case statement: on PLAG066 the `eta=1.5` rows have a spectral-norm error about 4% above the validated operator's. Only the first row leaves the validated approximation unchanged (pass-through is an exact change of representation); every other row is a different approximation and needs the application's own validation.

| Goal | Settings | PLAG066 / PLAG036 MB |
|---|---|---|
| Validated operator, exact representation changes only | `assemble_h2matrix_adaptive(K; rtol=1e-10, ...)`, then `H2PacketMatvecPlan(h2; workers=4, consume=true, passthrough=true)` | 288.1 / 661.0 |
| Smallest Float64 operator | `eta=1.5` (`adm=StrongAdmissibilityStd(1.5)`), `rtol=1e-10`; plan options `passthrough=true, coupling_rtol=2e-11, coupling_scale=:global` | 258.5 / 628.5 |
| Smallest operator overall | `eta=1.5`, `rtol=5e-11`; plan options `passthrough=true, coupling_rtol=1e-11, coupling_scale=:global, coupling_precision=Float16` | 179.1 / 474.9 |
| Reduced precision with a rigorous bound | `eta=1.5`, `rtol=1e-10`; `H2MixedPacketMatvecPlan(h2; workers=4, precision_rtol=1e-13, format48=true, passthrough=true, consume=true)` | 211.4 / 517.2 |
| Fastest single products | as above without `format48` | 245.9 / 578.5 |

In Merrill's end-to-end check on PLAG066 at 20 °C (six magnetization states against the dense operator), the largest tangent-torque differences were 6.95e-10 T for the validated settings and 4.1e-10, 2.5e-10 and 4.2e-10 T for the smallest Float64, smallest overall and rigorous-bound settings, and the largest energy differences 8.3e-8, 4.5e-8, 1.4e-7 and 4.5e-8 kT. These are single-grain checks, not a substitute for the validation steps below.

## Validate before enlarging a campaign

A practical sequence is:

1. Compare forward and adjoint products with a trusted reference, including rectangular shapes and row/column ordering where relevant.
2. Compare physical energy, field and tangent-torque errors on uniform/random states, minima and path images. Include temperatures/material states used in production.
3. Check an energy directional derivative against a finite difference of the same stored operator.
4. Run paired LEM and NEB solves from common seeds, endpoints and initial paths; evaluate both results with the same reference operator.
5. Check maximum and RMS residuals, barriers, unit-spin constraints and endpoints independently of the solver's convergence flag.
6. Measure retained memory, peak construction RSS and repeated-product/solver timing separately before increasing grain size or worker count.

The package has a [runnable rectangular reference example](assets/accuracy_and_plans.jl), requiring no micromagnetic meshes or caches:

```sh
julia --project=. --threads=4 docs/src/assets/accuracy_and_plans.jl
```

It demonstrates dense-reference conversion checks, exact stored adjoints, compact/packet products and independent caller copies. It does not substitute for application-level grain validation. The full PLAG066 protocol and downloadable evidence are on the [validation page](validation.md).
