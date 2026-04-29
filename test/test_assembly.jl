@testset "H² assembly and matvec" begin
    @testset "Chebyshev 2D Laplace" begin
        n = 180
        K, rowtree, coltree = separated_laplace2d(n; seed=42)
        h2 = assemble_h2matrix(K, rowtree, coltree; order=4, global_index=true)
        dense = dense_kernel_matrix(K, n)

        @test size(h2) == (n, n)
        @test H2Matrices.block_stats(h2).leaves > 0

        Random.seed!(10)
        err = H2Matrices.relative_matvec_error(h2, dense; nsamples=3)
        @test err < 5e-2

        summary = H2Matrices.compression_summary(h2)
        @test summary.compressed_bytes == H2Matrices.storage_bytes(h2)
        @test summary.dense_bytes == H2Matrices.dense_storage_bytes(h2)
        @test summary.blocks.leaves == length(H2Matrices.leaves(h2))
    end

    @testset "Chebyshev 3D Laplace" begin
        Random.seed!(123)
        n = 120
        src = [Point3D(rand(), rand(), rand()) for _ in 1:n]
        tgt = [Point3D(3.0 + rand(), rand(), rand()) for _ in 1:n]
        K = KernelMatrix(src, tgt) do x, y
            r = norm(x - y)
            r > 0 ? 1 / (4π * r) : 0.0
        end
        rowtree = ClusterTree(deepcopy(src), GeometricSplitter(; nmax=16))
        coltree = ClusterTree(deepcopy(tgt), GeometricSplitter(; nmax=16))
        h2 = assemble_h2matrix(K, rowtree, coltree; order=3, global_index=true)
        dense = dense_kernel_matrix(K, n)

        @test size(h2) == (n, n)
        @test H2Matrices.relative_matvec_error(h2, dense; nsamples=3) < 8e-2
    end

    @testset "mul! and dense conversion consistency" begin
        n = 90
        K, rowtree, coltree = separated_laplace2d(n; seed=9)
        h2 = assemble_h2matrix(K, rowtree, coltree; order=3, global_index=true)

        x = randn(n)
        y = zeros(n)
        mul!(y, h2, x)
        y2 = h2 * x
        @test norm(y - y2) < 1e-12

        y_prev = copy(y)
        mul!(y, h2, x, 2.0, 1.0)
        @test norm(y - (y_prev + 2.0 * y2)) < 1e-10

        M = Matrix(h2)
        @test norm(h2 * x - M * x) < 1e-10 * max(norm(M * x), eps())
    end
end
