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
        @test norm(Matrix(h2) - dense) / norm(dense) < 5e-6
        @test H2Matrices.storage_bytes(h2) > 0
        @test H2Matrices.dense_storage_bytes(h2) / H2Matrices.storage_bytes(h2) ≈ H2Matrices.compression_ratio(h2)
    end

    @testset "Adaptive convenience assembly" begin
        n = 140
        K, rowtree, coltree = separated_laplace2d(n; seed=23)
        h2 = assemble_h2matrix_adaptive(K, rowtree, coltree; rtol=1e-6, maxrank=40)
        dense = dense_kernel_matrix(K, n)
        @test norm(Matrix(h2) - dense) / norm(dense) < 5e-6
    end

    @testset "Threaded adaptive assembly and compressor override" begin
        n = 260
        K, rowtree, coltree = separated_laplace2d(n; seed=25)
        x = randn(MersenneTwister(26), n)
        serial = assemble_h2matrix_adaptive(K, rowtree, coltree; rtol=1e-7, maxrank=40)
        threaded = assemble_h2matrix_adaptive(K, rowtree, coltree; rtol=1e-7, maxrank=40,
                                              threads=true)
        # Leaves are assembled independently, so threading changes nothing.
        @test Matrix(threaded) == Matrix(serial)
        @test threaded * x == serial * x

        # An explicit compressor replaces the default PartialACA(rtol=aca_rtol).
        calls = Threads.Atomic{Int}(0)
        aca = PartialACA(; rtol=1e-8)
        counting = (K, rtree, ctree, buf=nothing) -> (Threads.atomic_add!(calls, 1); aca(K, rtree, ctree, buf))
        custom = assemble_h2matrix_adaptive(K, rowtree, coltree; rtol=1e-7, maxrank=40,
                                            comp=counting, threads=true)
        default = assemble_h2matrix_adaptive(K, rowtree, coltree; rtol=1e-7, maxrank=40,
                                             aca_rtol=1e-8)
        @test calls[] > 0
        @test Matrix(custom) == Matrix(default)

        # The AbstractKernelMatrix convenience method forwards both options.
        auto_serial = assemble_h2matrix_adaptive(K; rtol=1e-7, maxrank=40, nmax=24)
        auto_threaded = assemble_h2matrix_adaptive(K; rtol=1e-7, maxrank=40, nmax=24,
                                                   threads=true)
        @test Matrix(auto_threaded) == Matrix(auto_serial)
        dense = dense_kernel_matrix(K, n)
        @test norm(Matrix(auto_threaded) - dense) / norm(dense) < 1e-6
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
