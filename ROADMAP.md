# H2Matrices.jl Roadmap

This roadmap captures the next practical steps for turning the current H2Lib-inspired
Julia implementation into a clean, tested, solver-ready H²-matrix package.

## Immediate Priorities

1. Improve examples and documentation so they read like reproducible workflows,
   not just successful demo transcripts.
2. Add diagnostics utilities for storage, ranks, block counts, and approximation
   error estimates.
3. Reproduce the relevant H2Lib tests in Julia, starting with kernel assembly,
   H/H² conversion, recompression, and solver-facing matvec behavior.
4. Add solver wrappers around H² matvecs, because later demag matrix problems
   will need robust iterative solves.
5. Build true H² arithmetic and factorization in this package, not in a separate
   downstream package.

Micromagnetics-specific kernels and demag assembly are intentionally deferred
until the core H² matrix, diagnostics, tests, and solver path are sturdier.

## Documentation And Examples

- Refactor `docs/src/examples.md` into a clearer progression:
  - minimal 2D kernel example,
  - visual anatomy of dense/H/H² storage,
  - Chebyshev vs adaptive ACA assembly,
  - recompression accuracy/storage tradeoff,
  - large 3D kernel workflow,
  - solver-facing workflow.
- Replace single-entry spot checks with reproducible sampled or dense-reference
  matvec error estimates.
- Add tables for parameter choices: `order`, `rtol`, `maxrank`, `nmax`, and
  admissibility.
- Move repeated plotting and measurement code from doc scripts into reusable
  package/test helpers where appropriate.

## Diagnostics

Add a diagnostics layer with:

- compressed and dense storage estimates,
- row/column basis storage estimates,
- block statistics for admissible and dense leaves,
- rank statistics,
- sampled Frobenius and matvec error estimates,
- dense-reference helpers for small reproducible tests.

These utilities should support both documentation and tests.

## H2Lib Test Parity

The first H2Lib tests to mirror are:

- `test_kernelmatrix.c`
  - Newton, logarithmic, and exponential kernels,
  - dense reference comparison,
  - H² assembly and recompression.
- `test_h2compression.c`
  - H² matvec against dense reference,
  - weight computation,
  - local row/column weight consistency,
  - recompression and projection,
  - H-matrix to H² conversion.
- Krylov solver tests
  - CG and GMRES on dense reference problems,
  - CG and GMRES through H² `mul!`,
  - preconditioner hooks.

The H2Lib Cholesky/LR factorization tests should come after the H² arithmetic
layer is in place.

## Solver Path

Add package-native iterative solver wrappers:

- Conjugate gradient for SPD-compatible operators.
- Restarted GMRES for nonsymmetric operators.
- Optional left preconditioners.
- Return structured convergence information: iteration count, residual history,
  and convergence flag.

The solvers should work with any `AbstractMatrix`, but H² matrices are the main
target. This keeps later demag solves close to the compressed operator rather
than forcing dense assembly.

## H² Arithmetic And Factorization

True H² arithmetic and factorization belong in this package. The intended path is:

1. Add reliable copy/clone/project utilities for cluster bases and H² matrices.
2. Implement H² addition and scaled addition with recompression.
3. Implement H²-H² multiplication with truncation and recompression.
4. Add triangular solve/evaluation on hierarchical block structures.
5. Implement approximate LR and Cholesky factorizations.
6. Reproduce H2Lib `test_h2matrix.c` factorization checks against dense
   references on small problems.

This should be done deliberately, with tests at every stage, because the
factorization layer will become a foundation for large demag matrix solves.

## Suggested Implementation Order

1. Roadmap, docs cleanup, and diagnostics.
2. Test split and H2Lib kernel/compression parity.
3. Solver wrappers and solver tests.
4. Copy/project/recompression APIs.
5. H² arithmetic.
6. H² triangular solves and approximate factorizations.
