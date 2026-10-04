# Changelog

## 0.2.0

Faster and lower-memory construction, a new packet product engine, exact and
bounded storage reductions, and opt-in absolute error control. Default
tolerances are unchanged (`error_control=:block`, `aca_rtol = rtol/10`), but
the stored operator, the plan types and some defaults change, so this is a
breaking (0.x minor) release. 5,206 package tests pass on Julia 1.13.1
(537 in v0.1.3).

### Breaking changes

- **The conversion builds a different (equivalent) representation.**
  `compress_hmatrix_to_h2`, and therefore `assemble_h2matrix_adaptive`, use the
  condensed builder below. The operator equals the v0.1.3 construction up to
  rounding, but basis, transfer and coupling entries differ: untruncated bases
  are stored as identity leaves or identity embeddings instead of SVD bases,
  and ranks can differ where a singular value sits at the truncation threshold
  within rounding (e.g. exactly rank-deficient active sets with `rtol=0`).
  Stored operators are therefore not bitwise reproducible across 0.1 and 0.2.
  `compress_hmatrix_to_h2(...; _reference=true)` keeps the v0.1.3 builder for
  validation (it does not support `atol`).
- **Threaded conversion by default.** `compress_hmatrix_to_h2(...; threads)`
  and `assemble_h2matrix_adaptive(...; conversion_threads)` default to `true`
  when Julia has several threads and BLAS uses one. Results are bitwise
  independent of the thread count, but the conversion now spawns Julia tasks;
  pass `false` where the caller already uses every core.
- **Consumed intermediates.** `assemble_h2matrix_adaptive` now releases its
  private H-matrix block by block during the conversion.
  `compress_hmatrix_to_h2(H; consume=true)` (opt-in) leaves `H` unusable, also
  when the conversion throws.
- **Packet plan internals.** `H2PacketMatvecPlan` is now the parametric
  `H2PacketMatvecPlan{C}` (coupling-packet type `C`), with a new packet layout
  (near-field column packets, per-coupling slots, no per-worker reduction
  buffers). Code that read plan fields or dispatched on the concrete type must
  use `<:H2PacketMatvecPlan`; the internal `TransposedPacketH2Plan` alias is now
  `Union{Transpose{Float64,<:H2PacketMatvecPlan},Adjoint{Float64,<:H2PacketMatvecPlan}}`.
  Packet products are bitwise independent of the worker count and differ from
  v0.1.3 packet products only by rounding (at most 8.4e-16 relative, measured).
- **Factorized couplings stay factored.** Packet plans built with
  `coupling_rtol` keep `S = L*R'` as factors (`keep_factors=true`, default)
  instead of multiplying them out; `keep_factors=false` restores the v0.1.3
  packing.
- `assemble_h2matrix_adaptive(...; comp)` replaces the default
  `PartialACA(; rtol=aca_rtol)` compressor; `aca_rtol` is then ignored.

### Faster construction

- Condensed conversion: each cluster condenses its active set (direct
  partner-weighted factors plus inherited ancestor factors) by an exact
  Gram-preserving QR, so widths stay at most the cluster size instead of
  accumulating every ancestor block. Tall transfers are QR-condensed before
  their SVD, explicit rotations use `ormqr`, and a rigorous triangular-inverse
  certificate skips transfer SVDs at saturated parents.
- `threads`: row and column bases, independent subtrees and couplings run as
  Julia tasks; couplings are formed as soon as both of their bases are final,
  while the top-level basis work continues.
- `assemble_h2matrix_adaptive(...; threads=true)` threads HMatrices' leaf
  assembly (identical H-matrix; `K` must allow concurrent `getblock!`), and
  `comp` selects the H-block compressor.
- Measured on the campaign's PINT grains at the validated settings
  (`eta=3`, `rtol=1e-10`, `aca_rtol=1e-11`, four Julia threads, one BLAS
  thread, separate processes on a 14-core M4 Pro under load 2-5): the
  conversion took 1.39 / 8.97 / 24.2 / 31.8 s on PLAG066 / PLAG036 / PLAG022 /
  PLAG012 (6,028 / 12,415 / 17,875 / 30,321 boundary nodes), against 12.5 /
  50.7 / 137.8 / 257.8 s with v0.1.3. The sampled peak live heap of the whole
  build fell from 1.3-1.4 / 3.3-3.6 / 7.0 / 14.1-15.1 GB to 0.62 / 1.85 / 3.95 /
  6.0 GB (these builds also include Merrill's faster boundary kernel for the
  ACA stage, so only the conversion times isolate this package).

### Faster products

- One packet engine: couplings remain row packets, near-field blocks are split
  at elementary column intervals into column packets, and fused long-column
  kernels replace gathered BLAS GEMV in both directions. The upward pass,
  forward near field and downward pass run in parallel with explicit write
  ownership (transposed couplings and forward near-field packets write private
  slots that the owning downward-pass task reduces in fixed order), and forward
  products start a row subtree's downward pass as soon as its last coupling
  packet completes. Stored numeric data is unchanged.
- Four-worker forward/adjoint medians against the v0.1.3 packet plan in the
  same process: 1.83/1.62 against 3.42/2.75 ms (PLAG066), 5.01/4.71 against
  9.46/8.02 ms (PLAG036), 10.72/9.63 against 19.41/18.51 ms (PLAG022); in
  separate processes 22.2/18.5 against 34.6/29.4 ms on PLAG012. Single products
  are close to memory-bandwidth bound, so more than 6-8 workers helped little.
  Threaded products allocate 5-8 KB of task objects per call; one-worker plans
  do not allocate.
- Several right-hand sides: `mul!(Y, plan, X)` and transpose/adjoint products
  with matrices stream the operator once per block of up to 16 vectors (nine
  vectors on PLAG066: 0.59/0.50 ms per vector, against 1.64/1.44 ms for single
  products). `multi_workspace_bytes` reports and `release_multi_workspace!`
  frees or shrinks the kept multi-RHS workspace.

### Smaller stored operators

- `consume=true` for `H2PacketMatvecPlan(h2; ...)`, `H2CompactMatvecPlan(h2;
  ...)` and `H2MixedPacketMatvecPlan(h2; ...)` releases raw H² blocks while the
  plan is built; the plans are identical to non-consuming builds.
- `passthrough=true` (compact, packet and mixed plans): internal basis nodes
  whose transfers barely compress and serve few couplings use their children's
  concatenated coefficients, folding their transfers into couplings and parent
  transfers. **Exact up to rounding** (products changed by at most 1.4e-15
  relative): 302.8 → 288.1 MB on PLAG066, 714.8 → 661.0 MB on PLAG036,
  1511.5 → 1400.0 MB on PLAG022, 3555 → 3405 MB on PLAG012.
- Coupling truncation (`coupling_rtol`) gains `coupling_scale` (`:block`,
  `:global` = largest stored block norm, or a number) and
  `coupling_precision=Float32`/`Float16` (small retained components stored in
  reduced precision with exact power-of-two column scales where needed,
  Float64 arithmetic, adjoint the exact transpose of the stored operator).
  These are **approximations**; the reduced-precision rounding is estimated,
  not bounded. Both options are rejected without `coupling_rtol`.
- `H2MixedPacketMatvecPlan(compact_or_h2; workers, precision_rtol=1e-13,
  format48=false)` and `precision_summary`: coupling packets rotated to their
  left singular basis store low-weight rows in Float32 (optionally a 48-bit
  format) under the **rigorous a priori bound**
  `‖Ã - A‖_F ≤ precision_rtol · η` on the reduced-precision rounding (`η` the
  Frobenius norm of the stored blocks). It runs on the packet engine (bitwise
  independent of the worker count, multi-RHS products, deterministic
  construction), composes with pass-through, and `precision_rtol=0` reproduces
  `H2PacketMatvecPlan` bitwise. Measured rounding is about 4e-14 relative,
  three orders below the compression error. Single products are about as fast
  as or faster than Float64 packets (memory bound); multi-RHS products were up
  to 25% slower per vector on the larger grains (compute bound; 47% for one
  PLAG066 adjoint case), and `format48=true` saves another 10-14% of the bytes
  at a decode cost.

### Error control

- Opt-in global (absolute) error control:
  `compress_hmatrix_to_h2(...; atol, safeguard_rtol)` keeps basis singular
  values above `max(rtol*σ₁, min(atol, safeguard_rtol*σ₁))`, and
  `assemble_h2matrix_adaptive(...; error_control=:global, scale)` uses the
  absolute ACA tolerance `aca_rtol*s` and the basis threshold `rtol*s`, with
  `s` from the new `estimate_operator_scale` (RMS row norm from 32 sampled
  kernel rows). Explicit `atol`/`aca_atol` are also accepted. Defaults are
  unchanged.
- On the campaign's boundary operator, global control selects nearly the same
  ranks as block-relative control at equal *product* error (0-1% storage
  difference). Its value is how the error scales with the grain: product
  errors stayed at 1.5-2.4e-11 from PLAG066 to PLAG022 with one setting, and
  in Merrill's end-to-end check below it gave lower tangent-torque errors per
  stored byte than tighter block-relative tolerances.

### Accuracy finding and recommended settings

Merrill's end-to-end check compares energy and tangent torque against the
dense boundary operator (gate: 1e-9 T and 1e-6 kT). The settings validated on
PLAG066 with v0.1.x (`eta=3`, block-relative `rtol=1e-10`,
`aca_rtol=1e-11`) pass on PLAG066 (6.95e-10 T at 20 °C) but **miss the gate on
larger grains**: 8.3e-9 T / 2.3e-6 kT on PLAG036, 3.6e-9 T / 1.2e-6 kT on
PLAG022 and 1.4e-8 T / 4.3e-5 kT on PLAG012 (20 °C, six states; 570 °C is about
five times easier). v0.1.3 gives the same numbers, so this is not a
regression. The relative product errors were only about 1e-10; the torque is
an absolute, node-wise quantity, its worst nodes are small-weight boundary
nodes (many of them corners of a single cell), and at fixed block-relative
tolerances the absolute error grows with the grain. The smaller variants above (`eta=1.5` with Float64 truncation,
Float16 tiers, or mixed packets at `rtol=1e-10`) also miss the torque gate on
PLAG036-PLAG012 at 20 °C (1.7e-9 to 1.6e-8 T) even though they meet the
validated product-error levels.

Settings that meet the gate on all four grains:

```julia
h2 = assemble_h2matrix_adaptive(K; nmax=32,
    error_control=:global, rtol=3e-12, aca_rtol=3e-13,
    maxrank=typemax(Int), strict=true,
    adm=HMatrices.StrongAdmissibilityStd(1.5))
plan = H2MixedPacketMatvecPlan(h2; workers=4, precision_rtol=1e-13,
                               passthrough=true, consume=true)
```

Over 54 states per grain (six standard plus 48 random smooth ones) at 20 °C
and 570 °C, the largest torque differences were 2.1e-11 / 8.8e-11 / 1.5e-10 /
1.6e-10 T and the largest energy differences 9.8e-9 / 6.5e-8 / 7.4e-8 /
1.4e-7 kT on PLAG066 / PLAG036 / PLAG022 / PLAG012 (worst at 20 °C), with
stored operators of 246.8 / 650.1 / 1262.6 / 3226.1 MB against 302.8 / 714.8 /
1511.5 / 3555.1 MB for the validated settings. Product errors fell from 7.7e-11
to 8.7e-13 relative on PLAG036 (four-vector maxima). Costs: builds took 47.7 /
124.7 / 219.5 s on PLAG036 / PLAG022 / PLAG012, 2.9-3.7 times the validated
settings on 0.2.0 (15.3 / 43.3 / 58.9 s) but below v0.1.3's builds at the
validated settings (92 / 231 / 425 s with its serial ACA); the process's
maximum RSS after the build was 6.0 / 8.4 / 10.2 GB against 4.3 / 6.3 /
7.4 GB (including 2.4-7.7 GB of mesh data and factorizations held before the
build), and the sampled live-heap upper bound of the PLAG012 build 15.8 GB
against 6.4 GB. An independent check with 150 new physical states and a
PLAG036 L-BFGS minimization (energies within 4e-8 kT of the dense run) agreed;
states constructed to maximize the error do break the gate on PLAG012 (smooth
polynomial fields with singular points 1.44e-6 kT; rough non-physical states
1.7e-8 T and 2.6e-5 kT), with the largest errors at the application's gauge
reference node, so these are sampled maxima, not bounds. These are measurements of one application and kernel on one
machine, not guarantees; validate the settings for other operators.

### Documentation

- New sections on the condensed conversion and on absolute versus
  block-relative error (*How the advances work*), recommended settings for
  tight absolute accuracy (*Accuracy and performance*), and the 0.2.0
  four-grain results with downloadable CSVs (*Validation*). The PLAG066 page is
  kept as the v0.1.x evidence.

### Caveats on the quoted numbers

Product errors are the largest relative forward/adjoint errors against exact
dense products over four reference vectors (three Gaussian, one smooth); smooth
inputs and single columns see errors 1.5-4.5x larger, so they are comparative
levels, not bounds. Only implicit identities, packing and pass-through are
exact changes of representation; `eta`, tolerance, coupling-truncation and
precision options are different approximations that need application
validation. Timings are medians on a shared 14-core M4 Pro and varied by about
10%. Retained operator storage is not peak construction memory.

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
