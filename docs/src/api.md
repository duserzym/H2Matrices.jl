# API Reference

## Assembly

```@docs
assemble_h2matrix
assemble_h2matrix_adaptive
```

## Compression

```@docs
compress_hmatrix_to_h2
recompress!
compress_matrix_to_h2
```

## Diagnostics

```@docs
compression_summary
storage_bytes
dense_storage_bytes
block_stats
rank_stats
relative_matvec_error
sampled_frobenius_error
```

## Matrix–Vector Product

```@docs
h2matvec!
forward_transform!
backward_transform!
```

## Solvers

```@docs
H2SolveResult
solve_cg
solve_gmres
```

## Types

```@docs
ClusterBasis
H2Matrix
UniformBlock
```
