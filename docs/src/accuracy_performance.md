# Accuracy and performance

Adaptive assembly uses ACA followed by nested-basis conversion. Set the ACA
and basis tolerances independently, and use strict rank-cap checks:

```julia
using H2Matrices, LinearAlgebra
h2 = assemble_h2matrix_adaptive(K;
    rtol=1e-10, aca_rtol=1e-11, maxrank=512, strict=true)
BLAS.set_num_threads(1)
plan = H2PacketMatvecPlan(h2; workers=4)
mul!(y, plan, x)
mul!(z, adjoint(plan), y)
```

These parameters are an example, not universal accuracy requirements.
A local truncation tolerance is not a certificate of total field or solver
error. Check independent forward/adjoint products and physical observables
against a trusted reference. Tightening tolerance cannot overcome a rank cap.

## Reusable representations

`H2MatvecPlan` shares the source operator's matrices and reuses scratch.
`H2CompactMatvecPlan` replaces saturated bases with implicit identity bases,
keeping their action in couplings and transfers. No orthogonality assumption
or truncation is required. `H2PacketMatvecPlan` packs interactions into
contiguous matrices; packing changes only floating-point summation order.
These plans are matvec-only operators.

The packet plan defaults to one worker. Multiple workers use private adjoint
reduction buffers and fixed reduction order. Near-field forward products
fall back to serial execution if output row ranges overlap. Set BLAS to one
thread when using packet workers, and measure your own grain/workload.
Thread scheduling incurs small allocations; the single-worker warmed path
is allocation-free on tested recent Julia compilers.

Plans contain mutable scratch and are not safe for concurrent calls.
`copy(plan)` creates independent scratch, including factorized-coupling and
packet buffers, while sharing numerical matrices. Never mutate shared data
or plan metadata while workers are running. Rebuild plans after recompression
or changes to the source matrix.

`H2LowRankMatvecPlan` and the compact plan's `coupling_rtol` option introduce
an additional local SVD approximation. Those options require a separate error
budget and application validation. They remain opt-in.

## Measured campaign grain

PLAG066 is a real PINT campaign mesh with 19,901 nodes, 100,602 tetrahedra,
and 6,028 boundary nodes. At 570 °C, a four-worker packet plan retaining the
original ACA/basis tolerances (1e-11/1e-10) used 302.75 MB of numeric storage,
versus 337.78 MB for H2 and 378.59 MB for H. Forward/adjoint products measured
3.36/2.61 ms. In paired single runs, NEB polishing took 52.2 s versus 53.9 s
for H, with the same 345 iterations. LEM also retained its iteration count.

Maximum H-reference torque discrepancy was 5.5e-11 T on newly generated LEM
and NEB states. The barrier differed by 2.38e-10 kBT, and both maximum and RMS
NEB residual checks passed. Fixed-state checks at 25 °C, 400 °C, and 570 °C
cover 96 temperature/state combinations. These results validate this grain
and path, not mesh refinement, all minima, or a whole temperature campaign.

Numeric storage excludes workspace and object overhead. The selected plan's
Julia `summarysize` is 307.57 MB. Construction still temporarily holds H and
H2 data, with additional packet packing buffers; final savings do not establish
peak RSS savings. Separate worker copies share matrix data, reducing retained
memory across independent callers.
