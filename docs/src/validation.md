# Validation and measured results

This page has two parts. The [0.2.0 section](#0.2.0:-four-grains-and-the-torque-gate) checks four grains against the dense operator and records that the v0.1.x tolerances are not enough on the larger ones. The rest of the page is the **v0.1.x evidence**: the PLAG066 validation that selected those tolerances, kept unchanged as the record of that release.

## 0.2.0: four grains and the torque gate

Merrill's end-to-end check evaluates the micromagnetic energy and its gradient with the H² boundary operator and with the dense boundary operator, at the same magnetization states, and reports the largest nodal tangent-torque difference (T) and the energy difference (kT). The gates are `1e-9 T` and `1e-6 kT`. The states are six standard ones (two uniform, a vortex, two smooth random fields and random nodal directions) plus, where noted, 48 random smooth ones; temperatures 20 °C and 570 °C (20 °C is about five times harder). Grains PLAG066, PLAG036, PLAG022 and PLAG012 come from the same campaign as below, with 6,028, 12,415, 17,875 and 30,321 boundary nodes (dense operators 0.29, 1.23, 2.56 and 7.35 GB).

| Grain | v0.1.x settings, 20 °C, 6 states | Recommended 0.2.0 settings, 20 °C, 54 states | Recommended, 570 °C, 54 states | Stored MB (v0.1.x → recommended) |
|---|---|---|---|---|
| PLAG066 | 6.95e-10 T, 8.3e-8 kT | 2.1e-11 T, 9.8e-9 kT | 4.2e-12 T, 1.4e-10 kT | 302.8 → 246.8 |
| PLAG036 | **8.3e-9 T, 2.3e-6 kT** | 8.8e-11 T, 6.5e-8 kT | 1.8e-11 T, 9.1e-10 kT | 714.8 → 650.1 |
| PLAG022 | **3.6e-9 T, 1.2e-6 kT** | 1.5e-10 T, 7.4e-8 kT | 3.0e-11 T, 1.0e-9 kT | 1511.5 → 1262.6 |
| PLAG012 | **1.4e-8 T, 4.3e-5 kT** | 1.6e-10 T, 1.4e-7 kT | 3.3e-11 T, 1.9e-9 kT | 3555.1 → 3226.1 |

Bold entries miss a gate. The v0.1.x settings (`eta=3`, block-relative `rtol=1e-10`, `aca_rtol=1e-11`, Float64 packets) are the ones validated on PLAG066 below; v0.1.3 gives the same numbers, so this is not a regression. On the three larger grains their relative product errors stayed between 5.9e-11 and 1.3e-10; the torque is a node-wise absolute quantity whose worst values sit at small-weight boundary nodes, and block-relative truncation lets the absolute error grow with the grain. The recommended settings are `error_control=:global` with `rtol=3e-12` and `aca_rtol=3e-13`, `eta=1.5`, strict uncapped ranks, and `H2MixedPacketMatvecPlan` with `precision_rtol=1e-13` and `passthrough=true`; [Accuracy and performance](accuracy_performance.md#Recommended-settings-for-tight-absolute-accuracy) explains the choice, and [How the advances work](advances.md#Accuracy-at-scale:-absolute-rather-than-block-relative-error) the cause.

Costs: energy-and-gradient evaluations took the same time, but builds took 2.9-3.7 times longer on the three larger grains (47.7, 124.7 and 219.5 s against 15.3, 43.3 and 58.9 s), and the build needed more memory (maximum RSS after the PLAG012 build 10.2 GB against 7.4 GB; sampled live-heap upper bound 15.8 GB against 6.4 GB, similar to v0.1.3's 15.1 GB at the v0.1.x settings).

These are checks of one application's energy on four grains on one machine (14-core M4 Pro, 24 GB, Julia 1.13.1, four threads, one BLAS thread). An independent check added 150 new physical states (all within 1.6e-10 T and 1.1e-7 kT) and a 100-iteration L-BFGS minimization on PLAG036, whose minimum matched the dense run to 4e-8 kT; no NEB solves were repeated. States constructed to maximize the error break the gate on PLAG012 (see [the limits](accuracy_performance.md#Recommended-settings-for-tight-absolute-accuracy)), so these are sampled maxima, not bounds, and they do not certify other operators. Download the [end-to-end results](assets/validation/v020_end_to_end.csv) and the [build and product timings](assets/validation/v020_builds.csv) (hashes in the [SHA-256 ledger](assets/validation/sha256.csv)). The 0.2.0 package passes 5,206 tests on Julia 1.13.1.

## v0.1.x: PLAG066 validation

The corrected and optimized H² operator agrees with the H reference on one real PINT grain, one LEM seed and one nine-image inversion path. The selected representation retains ACA/basis tolerances `1e-11/1e-10`, without additional coupling truncation. Its gains are measured retained storage and product throughput; peak construction memory and larger-grain scaling remain separate questions.

### Input provenance

| Item | Recorded value |
|---|---|
| Mesh | `PLAG066.msh` |
| Campaign repository | `PINT_with_reversal` |
| Campaign branch | `origin/campaign/desktop-runs` |
| Inspected campaign tip | `ea8292ff10179cc0719b0c680a5836068239d048` |
| Mesh source commit in its history | `20548fd2fb3d908853fa62cdef3769b6441580af` |
| Source path | `data/Nikolaisen2022_merrill_msh/PLAG066.msh` |
| Mesh SHA-256 | `0dd36b374dd2c558a5269ec09d72f289c87d5a0903b182684976f89ba75c591d` |
| FEM nodes / tetrahedra | 19,901 / 100,602 |
| Boundary nodes | 6,028 |

The mesh was recovered from campaign history because it had been removed from the inspected tip. Its hash matches the recorded campaign mesh identity. Original metre coordinates were retained; the PINT checkout, campaign queues and outputs were unchanged. The compressed boundary matrix is 6,028 × 6,028, not 19,901 × 19,901.

### Reference and candidate construction

The H reference uses ACA tolerance `1e-11`, rank cap 512 and leaf size 32. Corrected H² is converted from that same H matrix with basis tolerance `1e-10` and strict rank-cap checks. This comparison isolates nested conversion error relative to H; it is not an independent continuum or mesh-refinement oracle.

The selected candidate is `H2PacketMatvecPlan(h2; workers=4)`, with no `coupling_rtol`. Its compact stage replaces saturated bases algebraically. Selection screens candidate field/energy/torque errors, enforces storage no larger than the H² baseline, and minimizes measured forward-plus-adjoint time among the screened choices. Leaf-32 and a previously screened leaf-256 representation were compared with 1, 2 and 4 workers. This is a small parameter screen, not an exhaustive optimum.

Shared FEM factors and sparse exchange isolate the boundary-operator effect. The paired solver setup uses magnetite properties at 570 °C, zero applied field, identity anisotropy orientation and `K2=0`. Julia used four threads and BLAS one thread. LEM starts from uniform +x magnetization and uses Riemannian L-BFGS with AMG preconditioning, targeting 0.5 mT tangent torque.

NEB polishing uses common H-derived antipodal endpoints and the same saved H RMS-screen path. There are nine images, geodesic geometry, L-BFGS, a 2 mT maximum-torque stopping target and no rescue stage. The reported NEB time is polishing of this common path, not complete path discovery from arbitrary endpoints.

### Why solver stopping was checked independently

The initial RMS-only screen stopped after 204 iterations, but its largest nodal torque was about 25 mT, above the campaign's 5 mT maximum criterion. Polishing with a maximum-torque target brought both bands below 2 mT. The polished barrier was about 4.7% below the RMS-screen barrier.

This showed that apparent convergence could be misleading even before comparing compression choices. Independent reconstruction checks maximum and RMS residuals, including the climbing image. Both H and H² results are evaluated with the same H reference, so a candidate cannot pass simply because its own approximate force is small.

### Storage and repeated products

All storage below is in decimal MB and counts numeric data only.

| Representation | Numeric MB | Forward median, ms | Adjoint median, ms | Forward / adjoint allocated bytes |
|---|---:|---:|---:|---:|
| H reference | 378.59 | 3.43 | 3.44 | 2,132,144 / 2,334,288 |
| Corrected H², generic traversal | 337.78 | 8.98 | 8.00 | 849,296 / 1,062,560 |
| Reusable H² baseline | 337.78 | 7.79 | 6.92 | 0 / 0 |
| Compact packet, 1 worker | 302.75 | 6.27 | 5.50 | 0 / 0 |
| Compact packet, 2 workers | 302.75 | 4.22 | 3.42 | 1,312 / 2,624 |
| Compact packet, 4 workers | **302.75** | **3.36** | **2.61** | **2,336 / 4,672** |

The selected operator saves **10.37%** versus corrected H² and **20.03%** versus H. Relative to the allocation-free reusable baseline, forward/adjoint products are **2.32×/2.65× faster**. Packing alone helps; workers further improve the measured products. NEB has other FEM/solver costs, so its elapsed time does not improve by the matvec speedup factor.

Matvec timings are medians of nine batches of twenty products after warm-up. Paired solver timings are single runs. The original tight-tolerance H² NEB polish took about 84 s in an earlier run; cross-run comparisons are less controlled than the paired results below.

The compact stage uses 539 implicit row bases and 537 implicit column bases. The packet operator has 263 coupling packets and 448 near-field packets. Near-field forward rows overlap on this grain, so that part falls back to serial evaluation. The adjoint uses private worker reduction buffers. These timings describe the v0.1.3 packet plan; the 0.2.0 packet layout (see [How the Advances Work](advances.md)) runs all phases in parallel and stores the same numeric data.

### Paired LEM and NEB results

| Quantity | H reference | Selected packet H² |
|---|---:|---:|
| LEM iterations | 67 | 67 |
| LEM elapsed, s | 1.636 | 1.621 |
| LEM maximum torque evaluated with H, mT | 0.482552133 | 0.482552134 |
| NEB polish iterations | 345 | 345 |
| NEB polish elapsed, s | 53.894 | 52.159 |
| NEB maximum residual evaluated with H, mT | 1.941594411 | 1.941594427 |
| NEB RMS residual evaluated with H, mT | 0.054951634 | 0.054951637 |
| Barrier evaluated with H, J | `9.012353414848914e-19` | `9.012353414863080e-19` |

The timings establish comparable H and selected-H² NEB performance in this run, not a statistically established H² solver-speed advantage. Both retain 345 iterations and meet the independent maximum/RMS gates.

The LEM state RMS difference is `1.07e-10` using the recorded per-node normalization. The barrier discrepancy against the saved H benchmark is `2.38e-10 kBT`. This CSV comparison uses the original H benchmark; subtracting the two paired table entries instead gives a slightly different roundoff-level value. Maximum H-reference torque discrepancy across the newly generated selected LEM state and its NEB images is **`5.50e-11 T`**, below the `1e-9 T` compression-error gate.

### Accuracy checks and gates

| Check | Acceptance gate used | Selected result |
|---|---:|---:|
| Fixed-state maximum tangent-torque discrepancy | `1e-9 T` | `7.54e-11 T` |
| Fixed-state absolute energy discrepancy | `1e-6 kBT` | `4.33e-10 kBT` |
| Plan/source forward and adjoint relative errors | `1e-9` | `5.81e-16` / `2.78e-16` |
| Generated-state maximum tangent-torque discrepancy | `1e-9 T` | `5.50e-11 T` |
| Barrier discrepancy against saved H benchmark | `1e-6 kBT` | `2.38e-10 kBT` |
| H-evaluated NEB RMS / maximum residual | `1 mT` / `5 mT` audit gates | `0.05495 mT` / `1.94159 mT` |

The screen covers uniform/random states, both original LEM minima and all images of the two original bands. The final checks add the selected minimum and band. An energy directional finite difference at step `1e-3` has relative discrepancy `5.06e-9`; step selection and cancellation matter, so this is not a global gradient-error certificate.

Fixed-state tests at 25, 400 and 570 °C cover 32 states per temperature, **96 combinations** in total. Maximum torque discrepancies on Julia 1.12.6 are approximately `4.40e-11`, `4.40e-11` and `7.54e-11 T`. Repeating those checks on Julia 1.13.1 gives maximum `7.54e-11 T`. Unit-spin error is at most `3.33e-16` and endpoint error `1.44e-15`.

Compression-error torque gates compare candidate and reference at the same state. Solver residual gates measure how far a state is from equilibrium. These are different quantities; neither should be substituted for the other.

### How the 244 MB experiment differs

An earlier leaf-128 candidate used basis tolerance `1e-7` plus coupling SVD tolerance `1e-10`, reaching 244.40 MB. Its screened torque discrepancy was about `3.46e-7 T`, below that experiment's `1e-6 T` gate. It also passed paired LEM/NEB checks for the tested path.

The selected 302.75 MB plan keeps the original `1e-10` basis tolerance and passes the stricter `1e-9 T` gate. The 244 MB result is a separately validated accuracy/memory tradeoff. Recompression to `1e-6` gave about 215.72 MB but failed the earlier torque gate, illustrating why minimum storage alone was not the selection criterion.

### What the memory numbers exclude

Selected numeric storage is **302,752,824 bytes**. Julia `summarysize` including plan metadata/scratch is **307,573,776 bytes**. Neither is process RSS or total micromagnetic simulation memory. Retaining the reference, assembly operator and intermediate plans together uses more memory than retaining only the selected operator.

A dense 6,028² Float64 boundary matrix would use **290.69 MB**, less than even this selected compressed operator at these tight tolerances. This grain validates compression accuracy and the optimization; it does not establish a storage crossover versus dense matrices. Asymptotic H² gains require favorable ranks/block structure and larger-size measurements.

Peak construction still includes an intermediate H matrix, H² data, SVD workspaces and packing buffers. The current API does not provide direct low-peak-memory ACA-to-H² construction. Direct Chebyshev interpolation also cannot simply be substituted for Merrill's current node-based BEM entry evaluator: that evaluator requires actual boundary-node incident faces and does not support arbitrary interpolation points.

### Artifacts and reproduction

Download the small measured data directly:

- [Candidate screening](assets/validation/plag066_screen.csv), [paired solvers](assets/validation/plag066_solvers.csv), and [selection summary](assets/validation/plag066_summary.csv).
- [Julia 1.12.6 temperature/state checks](assets/validation/plag066_temperature_states.csv), [plan/constraint checks](assets/validation/plag066_release_checks.csv), and [energy derivative](assets/validation/plag066_energy_derivative.csv).
- [Julia 1.13.1 temperature/state checks](assets/validation/plag066_julia113_temperature_states.csv) and [plan/constraint checks](assets/validation/plag066_julia113_release_checks.csv).
- [SHA-256 ledger for these downloadable files](assets/validation/sha256.csv).

The same original timing CSVs are attached to the [v0.1.2 release](https://github.com/duserzym/H2Matrices.jl/releases/tag/v0.1.2). Measurements were made on Julia 1.12.6; the Julia 1.13.1 follow-up repeats accuracy checks, not the timing experiment.

Full micromagnetic reproduction uses the Merrill drivers `example/pint_h2_lem_neb_validation.jl`, `example/pint_h2_compact_validation.jl`, `example/pint_h2_packet_validation.jl` and `example/pint_h2_release_stress.jl`, together with the hash-pinned mesh, saved H reference, solver checkpoints and compact-screen cache. These are maintained in the local investigation workspace; the H² package release does not bundle the large caches or mesh. Run each phase with Julia threads=4 and BLAS threads=1. This website supplies the numerical evidence, not a claim that the full grain workflow can run from the H² package alone.

The package-only [rectangular reference script](assets/accuracy_and_plans.jl) is self-contained and provides a smaller reproducible check of the core APIs.

### Regression coverage and limits

The v0.1.2 release passes 537 H² package tests on Julia 1.13.1, locally and in CI. Relevant tests isolate inherited directions, ACA rescaling, strict cap failure, zero-rank-parent traversal, rectangular/indexed adjoints, alpha/beta behavior, nonorthogonal/overcomplete bases, independent caller scratch and overlapping near-field reductions. Targeted Merrill tests passed 109 checks during development; the published v0.1.2 integration check passes 17 H² operator/gradient tests on Julia 1.13.1. The full Merrill test suite was not run for this investigation.

The evidence covers this grain, sampled states/temperatures and one inversion path. It does not establish all minima, temperature-driven basin identity, whole-campaign accuracy, mesh/path-refinement convergence, large-grain peak memory, or universal worker scaling. Those require additional application measurements. The [practical guide](accuracy_performance.md) gives the recommended validation order before enlarging production calculations.
