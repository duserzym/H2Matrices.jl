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
| `nmax`, admissibility | Tree/block partitioning and resulting ranks/work | A mesh-independent best configuration |
| `threads=true` | Builds the intermediate H-matrix leaves on all Julia threads; the result equals the serial build (`K` must allow concurrent `getblock!`) | Any accuracy change |
| `comp` | Replaces the default `PartialACA(; rtol=aca_rtol)` H-block compressor | Accuracy of the substitute compressor |

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
```

Use `adjoint(plan)` for the adjoint of this same stored approximation. Do not independently compress a transposed kernel and assume it is identical. A useful verification is `dot(z, plan*x) ≈ dot(adjoint(plan)*z, x)`, with a tolerance appropriate for accumulated rounding.

Use one BLAS thread when evaluating packet-worker speedups. Multiple packet workers and multiple BLAS threads can oversubscribe the processor. `workers=1` is the default; measure worker counts on the target grain and hardware. A one-worker warmed plan can avoid per-call allocations, while threaded paths create small scheduling objects.

## Retain only the operator you need

A standalone packet plan's storage reduction does not reduce the total live memory if the caller also retains H, H² and intermediate plans. For a product-only workload, limit their lifetimes:

```julia
function build_operator(K)
    h2 = assemble_h2matrix_adaptive(K;
        rtol=1e-10, aca_rtol=1e-11,
        maxrank=512, nmax=32, strict=true)
    return H2PacketMatvecPlan(h2; workers=4)
end
plan = build_operator(K)
```

After returning, unused assembly objects can be reclaimed; reclamation is governed by Julia's garbage collector. Construction still needs the intermediate H matrix, basis/SVD workspaces and packet packing buffers. This API does not establish a smaller peak assembly RSS.

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

Copies share numerical matrices and traversal metadata, but own coefficient/vector scratch, factorized-coupling scratch and packet reduction buffers. Do not mutate shared data while any caller is running. When outer tasks already use the available CPU budget, one packet worker per caller can avoid excessive nested parallelism.

Rebuild plans after successful recompression or source changes. `copy(plan)` creates a new workspace for the same representation; it does not rebuild it against a changed source.

## Optional approximation reductions

```julia
compact = H2CompactMatvecPlan(h2; coupling_rtol=1e-10)
# Alternatively:
factored = H2LowRankMatvecPlan(h2; rtol=1e-10)
```

These options perform local coupling SVD truncation and require another accuracy check. Recompression and relaxed basis tolerances likewise need a physical error budget. Do not infer equivalent torque accuracy from equivalent storage size or from the same numerical tolerance at different stages.

Strict recompression is transactional:

```julia
recompress!(h2; rtol=1e-9, maxrank=512, strict=true)
# Rebuild any plans after success.
```

A rejected strict rank cap leaves `h2` unchanged. The temporary copy increases setup memory. Conversion and recompression traverse active children even when their parent has rank zero.

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
