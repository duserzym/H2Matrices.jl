@testset "Adaptive compression and recompression" begin
    @testset "H-matrix to H² conversion" begin
        n = 180
        K, rowtree, coltree = separated_laplace2d(n; seed=22)
        hmat = assemble_hmatrix(K, rowtree, coltree;
            comp=PartialACA(; rtol=1e-9),
            global_index=true,
            threads=false)

        h2 = compress_hmatrix_to_h2(hmat; rtol=1e-6, maxrank=40)
        dense = dense_kernel_matrix(K, n)
        @test H2Matrices.relative_matvec_error(h2, dense; nsamples=3) < 8e-2
        @test H2Matrices.storage_bytes(h2) > 0
        @test H2Matrices.dense_storage_bytes(h2) / H2Matrices.storage_bytes(h2) ≈ H2Matrices.compression_ratio(h2)
    end

    @testset "Adaptive convenience assembly" begin
        n = 140
        K, rowtree, coltree = separated_laplace2d(n; seed=23)
        h2 = assemble_h2matrix_adaptive(K, rowtree, coltree; rtol=1e-6, maxrank=40)
        dense = dense_kernel_matrix(K, n)
        @test H2Matrices.relative_matvec_error(h2, dense; nsamples=3) < 8e-2
    end

    @testset "In-place recompression" begin
        n = 180
        K, rowtree, coltree = separated_laplace2d(n; seed=24)
        h2 = assemble_h2matrix(K, rowtree, coltree; order=5, global_index=true)
        dense = dense_kernel_matrix(K, n)

        before = H2Matrices.rank_stats(h2).row.total
        original = copy(h2)
        recompress!(h2; rtol=1e-4, maxrank=30)
        after = H2Matrices.rank_stats(h2).row.total

        @test after <= before
        @test H2Matrices.relative_matvec_error(h2, dense; nsamples=3) < 3e-1
        @test H2Matrices.relative_matvec_error(original, dense; nsamples=3) < 1e-1
    end

    @testset "Dense matrix to H² projection" begin
        A, tree = dense_spd_problem(80; seed=31)
        h2 = H2Matrices.compress_matrix_to_h2(A, tree, tree;
            rtol=1e-8,
            maxrank=30,
            global_index=false)

        @test size(h2) == size(A)
        @test H2Matrices.relative_matvec_error(h2, A; nsamples=3) < 1e-6
        @test H2Matrices.block_stats(h2).leaves > 0
    end
end
