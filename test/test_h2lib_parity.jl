@testset "H2Lib parity kernels and weights" begin
    @testset "KernelMatrix variants from H2Lib" begin
        for kind in (:newton, :logarithmic, :exponential)
            pts = random_points2d(96; seed=100 + length(String(kind)))
            K = kernel_matrix(pts, kind)
            tree = ClusterTree(deepcopy(pts), GeometricSplitter(; nmax=16))
            h2 = assemble_h2matrix(K, tree, tree;
                order = kind === :exponential ? 3 : 4,
                global_index = true)
            dense = dense_kernel_matrix(K, length(pts))

            err = H2Matrices.relative_matvec_error(h2, dense; nsamples=3)
            tol = kind === :exponential ? 5e-2 : 2e-1
            @test err < tol

            h2r = copy(h2)
            rank_before = H2Matrices.rank_stats(h2r).row.total
            recompress!(h2r; rtol=1e-4, maxrank=25)
            rank_after = H2Matrices.rank_stats(h2r).row.total
            @test rank_after <= rank_before
            @test H2Matrices.relative_matvec_error(h2r, dense; nsamples=3) < max(tol, 3e-1)
        end
    end

    @testset "Recompression weight dimensions" begin
        n = 120
        K, rowtree, coltree = separated_laplace2d(n; seed=55)
        h2 = assemble_h2matrix(K, rowtree, coltree; order=4, global_index=true)

        row_basis_weights = H2Matrices._compute_basis_weights(h2.row_basis)
        col_basis_weights = H2Matrices._compute_basis_weights(h2.col_basis)
        row_local = H2Matrices._compute_local_weights(h2, col_basis_weights, :row)
        col_local = H2Matrices._compute_local_weights(h2, row_basis_weights, :col)
        row_total = H2Matrices._accumulate_total_weights(h2.row_basis, row_local)
        col_total = H2Matrices._accumulate_total_weights(h2.col_basis, col_local)

        for cb in H2Matrices.nodes(h2.row_basis)
            W = row_total[objectid(cb)]
            @test size(W, 2) == cb.k
        end
        for cb in H2Matrices.nodes(h2.col_basis)
            W = col_total[objectid(cb)]
            @test size(W, 2) == cb.k
        end

        h2copy = copy(h2)
        recompress!(h2copy; rtol=1e-4, maxrank=25)
        @test H2Matrices.storage_bytes(h2copy) <= H2Matrices.storage_bytes(h2)
    end
end
