# Changelog

## Unreleased

- Restructure `H2PacketMatvecPlan` for race-free parallel phases. Couplings
  remain row packets; near-field blocks are split at elementary column
  intervals into column packets. Fused long-column kernels replace gathered
  BLAS GEMV in both directions. Transposed couplings and forward near-field
  write private slots that the owning downward-pass task reduces in fixed
  order. The upward pass, forward near field and downward pass now run in
  parallel, per-worker reduction buffers are removed, and products are bitwise
  independent of the worker count. Stored numeric data is unchanged.
- Add `mul!(Y, plan, X)` and transpose/adjoint products with matrices for
  packet plans, streaming the operator once per block of up to 16 right-hand
  sides, and `multi_workspace_bytes`.

## 0.1.3

Documentation-only release: explain the causes of the accuracy fixes and the
compact/packet advances, publish the PLAG066 protocol and downloadable evidence,
and add a runnable rectangular reference example. Clarify rank-dependent scaling,
retained versus peak memory, and the separate relaxed-tolerance tradeoff. Numerical
source and defaults are unchanged.

## 0.1.2

- Require Julia 1.13 or later and test the latest stable Julia release in CI.
- Keep the validated numerical implementation and compression tolerances unchanged.

## 0.1.1

- Correct adaptive nested basis construction: retain inherited interactions,
  weight ACA factors by partner QR factors, and avoid inflated unused bases.
- Apply transpose/adjoint products using the same stored operator, supporting
  rectangular matrices and independent row/column permutations.
- Add strict assembly and transactional recompression rank-cap checks. Fix
  recompression traversal through zero-rank parents. Local tolerances do not
  certify global operator errors.
- Add reusable matvec workspaces, implicit saturated bases, and packed coupling
  and near-field products. Compact/packet representation changes do not relax
  tolerances; optional low-rank coupling truncation is separately controlled.
- Add threaded packet products with private adjoint reductions, safe overlapping
  near-field fallback, and cheap independent worker copies sharing numerical data.
- Add regression and concurrency tests plus Julia 1.8/current CI. Require
  HMatrices 0.2.13 or a later compatible 0.2 release. Julia 1.8 is the minimum
  because this HMatrices release uses constant fields in mutable structs.

A real PLAG066 campaign mesh validation at 570 °C retains ACA/basis tolerances
1e-11/1e-10. Four-worker packet storage is 302.75 MB, versus 337.78 MB for H2
and 378.59 MB for H. Paired NEB runs took 52.2 s (packet) and 53.9 s (H), with
H-reference maximum torque 1.94 mT and barrier difference 2.38e-10 kBT. These
are single-grain measurements; final storage is not peak construction memory.
Fixed-state operator checks also pass at 25 °C and 400 °C.

## 0.1.0

Initial package release.
