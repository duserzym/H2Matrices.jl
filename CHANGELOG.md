# Changelog

## Unreleased

- Faster, lower-memory H → H² conversion with the same operator up to rounding:
  each cluster condenses its active set (direct partner-weighted factors plus
  inherited ancestor factors) by an exact Gram-preserving QR, so widths stay at
  most the cluster size instead of accumulating every ancestor block. Truncation
  ranks are unchanged except for rounding-level decisions at the truncation
  threshold (e.g. exactly rank-deficient blocks with `rtol=0`). The v0.1.3
  builder remains available as `compress_hmatrix_to_h2(...; _reference=true)`
  for validation.
- Untruncated bases are stored as identity leaves or identity embeddings (an
  exact change of coordinates); a rigorous triangular-inverse certificate skips
  transfer SVDs at saturated parents; compact plans skip exact identity
  expansions (bitwise identical couplings).
- `compress_hmatrix_to_h2(...; threads)` builds row/column bases, independent
  subtrees and couplings with Julia tasks (default when Julia has several
  threads and BLAS uses one). Results are bitwise independent of the thread count.
- Couplings are formed as soon as both of their cluster bases are final, while
  the remaining (top-level) basis work continues, using otherwise idle threads.
- `consume=true` releases each H-matrix block as soon as its coupling exists
  (during the basis construction; the H-matrix is unusable afterwards, also if
  the conversion throws) and `H2PacketMatvecPlan(h2; consume=true)` releases
  raw H² blocks while packing. `assemble_h2matrix_adaptive` consumes its private
  H-matrix; its `threads` keyword threads the H-matrix assembly (`K` must
  allow concurrent `getblock!`), `conversion_threads` the conversion, and
  `comp` selects the compressor.
- Restructure `H2PacketMatvecPlan` for race-free parallel phases. Couplings
  remain row packets; near-field blocks are split at elementary column
  intervals into column packets. Fused long-column kernels replace gathered
  BLAS GEMV in both directions. Transposed couplings and forward near-field
  write private slots that the owning downward-pass task reduces in fixed
  order. The upward pass, forward near field and downward pass now run in
  parallel, per-worker reduction buffers are removed, and products are bitwise
  independent of the worker count. Stored numeric data is unchanged.
- One packet engine for several packet layouts: `H2PacketMatvecPlan{C}` is
  parametric in its coupling-packet type, and the structured packets below and
  the mixed-precision packets of `H2MixedPacketMatvecPlan` share its phases,
  write ownership, near-field column packets, slot reductions and kernels
  (`TransposedPacketH2Plan` now uses `<:H2PacketMatvecPlan`).
- `H2CompactMatvecPlan(h2; passthrough=true)`: internal basis nodes whose
  transfer matrices barely compress and serve few couplings use their
  children's concatenated coefficients; their transfers are folded into
  couplings and parent transfers (exact up to rounding; less storage and
  faster products). Packet plans copy those coefficients in the upward and
  downward passes.
- Compact-plan coupling truncation (`coupling_rtol`) gains `coupling_scale`:
  `:block` (default, each coupling's own norm), `:global` (the largest stored
  block norm, near field included) or a positive number `s` (threshold
  `coupling_rtol*s`), and `coupling_precision=Float32` or `Float16` (small
  retained components stored in reduced precision with exact power-of-two
  column scales where needed, Float64 arithmetic, so the adjoint stays the
  exact transpose of the stored operator). Both qualify `coupling_rtol` and
  are rejected without it. These are approximations.
- Packet plans keep factorized couplings as factors (`keep_factors=true`;
  `false` multiplies them out), store reduced-precision couplings in
  per-precision packet parts, and drop couplings without any retained
  component. `H2PacketMatvecPlan(h2; consume=true, compact options...)` and
  `H2CompactMatvecPlan(h2; consume=true)` release the source blocks while the
  plan is built (each transformed coupling once factorized); the plans are
  identical to non-consuming builds.
- Add `H2MixedPacketMatvecPlan(compact_or_h2; workers, precision_rtol=1e-13,
  format48=false)` and `precision_summary`: coupling packets rotated to their
  left singular basis store low-weight rows in Float32 (optionally a 48-bit
  format) under an a priori Frobenius bound `precision_rtol * η` on the
  reduced-precision rounding; products accumulate in Float64 and the adjoint
  is the exact transpose of the stored operator. It runs on the packet engine
  (near field in Float64 column packets), so products are bitwise independent
  of the worker count and support several right-hand sides; construction is
  bitwise deterministic under threads; packets whose norm could overflow a
  reduced format stay in Float64. It composes with pass-through bases: the
  rotations of pass-through nodes and of their children, which share
  coefficients with their parent, are stored as reflectors instead of being
  absorbed into the basis. From an H² matrix it forwards compact options
  (`passthrough`, `coupling_rtol`, `coupling_scale`) to a consuming compact
  plan with `consume=true`; factorized couplings are multiplied out, and
  `coupling_precision` (a different reduced-precision storage) is rejected
  before anything is consumed.
- Global (absolute) error control, opt-in: `compress_hmatrix_to_h2(...; atol,
  safeguard_rtol)` keeps basis singular values above
  `max(rtol*σ₁, min(atol, safeguard_rtol*σ₁))`;
  `assemble_h2matrix_adaptive(...; error_control=:global, scale)` uses the
  absolute ACA tolerance `aca_rtol*s` and basis threshold `rtol*s` with the
  new `estimate_operator_scale` (RMS row norm from sampled kernel rows), and
  also accepts explicit `atol`/`aca_atol`. Defaults are unchanged. On the
  campaign's boundary operator global control saved only about 1% over
  block-relative control at equal accuracy.
- Add `mul!(Y, plan, X)` and transpose/adjoint products with matrices for
  packet plans, streaming the operator once per block of up to 16 right-hand
  sides, `multi_workspace_bytes` and `release_multi_workspace!` (frees or
  shrinks the multi-RHS scratch kept by the plan).
- Measured on the campaign's PLAG066/PLAG036 boundary operators (6,028/12,415
  nodes; dense 290.7/1233 MB): the validated `eta=3` operator is 302.8/714.8
  MB; exact pass-through 288.1/661.0 MB; with `eta=1.5` and Float64 global
  coupling truncation 258.5/628.5 MB; with Float16 coupling tiers
  179.1/474.9 MB; mixed-precision packets on pass-through `eta=1.5` bases
  211.4/517.2 MB (`format48=true`) under the 1e-13 bound. All of these meet
  the validated product-error levels against exact dense products; see
  `docs/src/advances.md` for timings and the trade-offs.

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
