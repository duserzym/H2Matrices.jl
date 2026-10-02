# Changelog

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
