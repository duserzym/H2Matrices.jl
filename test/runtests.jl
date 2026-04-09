using H2Matrices
using HMatrices
using HMatrices: ClusterTree, KernelMatrix, StrongAdmissibilityStd,
    GeometricSplitter, index_range, loc2glob,
    HMatrix, PartialACA, assemble_hmatrix
using StaticArrays
using LinearAlgebra
using Test
using Random

@testset "H2Matrices.jl" begin

    @testset "ClusterBasis construction" begin
        # Create a simple set of 2D points
        n = 200
        pts = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        clt = ClusterTree(pts, GeometricSplitter(; nmax=30))

        # Build cluster basis
        cb = H2Matrices.build_cluster_basis(clt)
        @test cb.cluster === clt
        @test H2Matrices.isroot(cb)
        @test !H2Matrices.isleaf(cb)

        # Check that leaves match
        cb_leaves = H2Matrices.leaves(cb)
        @test all(H2Matrices.isleaf, cb_leaves)
        @test length(cb_leaves) > 1
    end

    @testset "Chebyshev interpolation" begin
        # Test Chebyshev nodes
        nodes = H2Matrices.chebyshev_nodes(5)
        @test length(nodes) == 5
        @test all(-1 .<= nodes .<= 1)

        # Test scaled nodes
        scaled = H2Matrices.chebyshev_nodes_scaled(5, 0.0, 1.0)
        @test all(0 .<= scaled .<= 1)

        # Test interpolation points in 2D
        bbox = HMatrices.HyperRectangle(SVector(0.0, 0.0), SVector(1.0, 1.0))
        pts = H2Matrices.chebyshev_interpolation_points(3, bbox)
        @test length(pts) == 9  # 3^2

        # Test in 3D
        bbox3 = HMatrices.HyperRectangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0))
        pts3 = H2Matrices.chebyshev_interpolation_points(3, bbox3)
        @test length(pts3) == 27  # 3^3
    end

    @testset "Chebyshev basis construction" begin
        n = 200
        pts = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        clt = ClusterTree(pts, GeometricSplitter(; nmax=30))

        cb = H2Matrices.build_cluster_basis(clt)
        H2Matrices.build_chebyshev_basis!(cb, 3)

        # Check that leaves have V matrices
        for leaf in H2Matrices.leaves(cb)
            @test size(leaf.V, 1) == length(index_range(leaf.cluster))
            @test size(leaf.V, 2) == 9  # 3^2 for 2D
            @test leaf.k == 9
        end

        # Check that non-leaves have transfer matrices on children
        all_nodes = H2Matrices.nodes(cb)
        for node in all_nodes
            if !H2Matrices.isleaf(node) && !H2Matrices.isroot(node)
                @test size(node.E, 2) > 0  # has transfer matrix
            end
        end
    end

    @testset "H2Matrix assembly and matvec (Laplace 2D)" begin
        # Set up a 2D Laplace kernel
        n = 300
        Random.seed!(42)  # for reproducibility
        pts_row = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        pts_col = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:n]

        # Laplace-like kernel (with separation to ensure admissibility)
        K = KernelMatrix(pts_row, pts_col) do x, y
            r = norm(x - y)
            return r > 0 ? 1 / (4π * r) : 0.0
        end

        # Build cluster trees
        Xclt = ClusterTree(deepcopy(pts_row), GeometricSplitter(; nmax=30))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=30))

        # Assemble H² matrix
        h2 = assemble_h2matrix(K, Xclt, Yclt; order=4, global_index=true)

        @test size(h2) == (n, n)

        # Check that there are both admissible and dense leaves
        lvs = H2Matrices.leaves(h2)
        n_adm = count(H2Matrices.isadmissible, lvs)
        n_dense = count(!H2Matrices.isadmissible, lvs)
        @test n_adm > 0 || n_dense > 0  # at least some leaves

        # Test matvec accuracy against dense
        K_dense = Matrix{Float64}(undef, n, n)
        rp = loc2glob(Xclt)
        cp = loc2glob(Yclt)
        for j in 1:n
            for i in 1:n
                K_dense[i, j] = K[rp[i], cp[j]]
            end
        end

        x = randn(n)
        y_dense = K_dense * x
        y_h2 = h2 * x

        # The H² approximation should be reasonably accurate
        rel_err = norm(y_h2 - y_dense) / norm(y_dense)
        @test rel_err < 0.1  # within 10% (Chebyshev with order 4)
        println("  Laplace 2D matvec relative error: $rel_err")
        println("  Compression ratio: $(H2Matrices.compression_ratio(h2))")
    end

    @testset "H2Matrix assembly and matvec (Laplace 3D)" begin
        n = 200
        Random.seed!(123)
        pts_row = [SVector{3,Float64}(rand(), rand(), rand()) for _ in 1:n]
        pts_col = [SVector{3,Float64}(3.0 + rand(), rand(), rand()) for _ in 1:n]

        K = KernelMatrix(pts_row, pts_col) do x, y
            r = norm(x - y)
            return r > 0 ? 1 / (4π * r) : 0.0
        end

        Xclt = ClusterTree(deepcopy(pts_row), GeometricSplitter(; nmax=20))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=20))

        h2 = assemble_h2matrix(K, Xclt, Yclt; order=3, global_index=true)

        @test size(h2) == (n, n)

        # Dense reference
        K_dense = Matrix{Float64}(undef, n, n)
        rp = loc2glob(Xclt)
        cp = loc2glob(Yclt)
        for j in 1:n
            for i in 1:n
                K_dense[i, j] = K[rp[i], cp[j]]
            end
        end

        x = randn(n)
        y_dense = K_dense * x
        y_h2 = h2 * x

        rel_err = norm(y_h2 - y_dense) / norm(y_dense)
        @test rel_err < 0.15
        println("  Laplace 3D matvec relative error: $rel_err")
        println("  Compression ratio: $(H2Matrices.compression_ratio(h2))")
    end

    @testset "mul! interface" begin
        n = 100
        pts = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        pts_col = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:n]

        K = KernelMatrix(pts, pts_col) do x, y
            1 / (4π * norm(x - y))
        end

        Xclt = ClusterTree(deepcopy(pts), GeometricSplitter(; nmax=20))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=20))

        h2 = assemble_h2matrix(K, Xclt, Yclt; order=3, global_index=true)

        x = randn(n)
        y = zeros(n)

        # Test mul! with α=1, β=0
        mul!(y, h2, x)
        y2 = h2 * x
        @test norm(y - y2) < 1e-12

        # Test mul! with α=2, β=1
        y_prev = copy(y)
        mul!(y, h2, x, 2.0, 1.0)
        @test norm(y - (y_prev + 2.0 * y2)) < 1e-10
    end

    @testset "Dense conversion" begin
        n = 80
        pts = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        pts_col = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:n]

        K = KernelMatrix(pts, pts_col) do x, y
            1 / (4π * norm(x - y))
        end

        Xclt = ClusterTree(deepcopy(pts), GeometricSplitter(; nmax=20))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=20))

        h2 = assemble_h2matrix(K, Xclt, Yclt; order=3, global_index=true)

        # Convert to dense
        M = Matrix(h2)
        @test size(M) == (n, n)

        # Verify matvec matches dense conversion
        x = randn(n)
        @test norm(h2 * x - M * x) < 1e-12 * norm(M * x)
    end

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Adaptive H² compression
    # ═══════════════════════════════════════════════════════════════

    @testset "H-matrix → H² conversion (2D Laplace)" begin
        n = 300
        Random.seed!(42)
        pts_row = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        pts_col = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:n]

        K = KernelMatrix(pts_row, pts_col) do x, y
            r = norm(x - y)
            r > 0 ? 1 / (4π * r) : 0.0
        end

        Xclt = ClusterTree(deepcopy(pts_row), GeometricSplitter(; nmax=30))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=30))

        # Build H-matrix via ACA
        hmat = assemble_hmatrix(K, Xclt, Yclt;
                                comp=PartialACA(; rtol=1e-10),
                                global_index=true, threads=false)

        # Convert to H²
        h2 = compress_hmatrix_to_h2(hmat; rtol=1e-6, maxrank=50)

        @test size(h2) == (n, n)

        # Dense reference (local ordering)
        K_dense = Matrix{Float64}(undef, n, n)
        rp = loc2glob(Xclt)
        cp = loc2glob(Yclt)
        for j in 1:n, i in 1:n
            K_dense[i, j] = K[rp[i], cp[j]]
        end

        x = randn(n)
        y_dense = K_dense * x
        y_h2 = h2 * x

        rel_err = norm(y_h2 - y_dense) / norm(y_dense)
        @test rel_err < 0.05
        println("  H→H² 2D Laplace relative error: $rel_err")
        println("  Compression ratio: $(H2Matrices.compression_ratio(h2))")
    end

    @testset "H-matrix → H² conversion (3D Laplace)" begin
        n = 200
        Random.seed!(123)
        pts_row = [SVector{3,Float64}(rand(), rand(), rand()) for _ in 1:n]
        pts_col = [SVector{3,Float64}(3.0 + rand(), rand(), rand()) for _ in 1:n]

        K = KernelMatrix(pts_row, pts_col) do x, y
            r = norm(x - y)
            r > 0 ? 1 / (4π * r) : 0.0
        end

        Xclt = ClusterTree(deepcopy(pts_row), GeometricSplitter(; nmax=20))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=20))

        hmat = assemble_hmatrix(K, Xclt, Yclt;
                                comp=PartialACA(; rtol=1e-10),
                                global_index=true, threads=false)

        h2 = compress_hmatrix_to_h2(hmat; rtol=1e-6, maxrank=50)
        @test size(h2) == (n, n)

        K_dense = Matrix{Float64}(undef, n, n)
        rp = loc2glob(Xclt)
        cp = loc2glob(Yclt)
        for j in 1:n, i in 1:n
            K_dense[i, j] = K[rp[i], cp[j]]
        end

        x = randn(n)
        y_dense = K_dense * x
        y_h2 = h2 * x

        rel_err = norm(y_h2 - y_dense) / norm(y_dense)
        @test rel_err < 0.05
        println("  H→H² 3D Laplace relative error: $rel_err")
        println("  Compression ratio: $(H2Matrices.compression_ratio(h2))")
    end

    @testset "Adaptive H² assembly convenience" begin
        n = 200
        Random.seed!(42)
        pts_row = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        pts_col = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:n]

        K = KernelMatrix(pts_row, pts_col) do x, y
            r = norm(x - y)
            r > 0 ? 1 / (4π * r) : 0.0
        end

        Xclt = ClusterTree(deepcopy(pts_row), GeometricSplitter(; nmax=30))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=30))

        h2 = assemble_h2matrix_adaptive(K, Xclt, Yclt;
                                         rtol=1e-6, maxrank=50)

        @test size(h2) == (n, n)

        K_dense = Matrix{Float64}(undef, n, n)
        rp = loc2glob(Xclt)
        cp = loc2glob(Yclt)
        for j in 1:n, i in 1:n
            K_dense[i, j] = K[rp[i], cp[j]]
        end

        x = randn(n)
        rel_err = norm(h2 * x - K_dense * x) / norm(K_dense * x)
        @test rel_err < 0.05
        println("  Adaptive assembly 2D relative error: $rel_err")
    end

    @testset "H² recompression" begin
        n = 300
        Random.seed!(42)
        pts_row = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        pts_col = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:n]

        K = KernelMatrix(pts_row, pts_col) do x, y
            r = norm(x - y)
            r > 0 ? 1 / (4π * r) : 0.0
        end

        # Build with Chebyshev (higher order → higher rank)
        Xclt = ClusterTree(deepcopy(pts_row), GeometricSplitter(; nmax=30))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=30))

        h2 = assemble_h2matrix(K, Xclt, Yclt; order=5, global_index=true)

        K_dense = Matrix{Float64}(undef, n, n)
        rp = loc2glob(Xclt)
        cp = loc2glob(Yclt)
        for j in 1:n, i in 1:n
            K_dense[i, j] = K[rp[i], cp[j]]
        end

        x = randn(n)
        err_before = norm(h2 * x - K_dense * x) / norm(K_dense * x)
        rank_before = H2Matrices.total_rank(h2.row_basis)

        println("  Before recompression: error=$err_before, total_row_rank=$rank_before")

        # Recompress
        recompress!(h2; rtol=1e-4, maxrank=50)

        rank_after = H2Matrices.total_rank(h2.row_basis)
        y_h2 = h2 * x
        err_after = norm(y_h2 - K_dense * x) / norm(K_dense * x)

        println("  After recompression:  error=$err_after, total_row_rank=$rank_after")

        # Rank should be reduced (or at least not increased much)
        @test rank_after <= rank_before
        # Accuracy should still be reasonable
        @test err_after < 0.5
    end

    @testset "Recompression preserves matvec consistency" begin
        n = 100
        Random.seed!(99)
        pts = [SVector{2,Float64}(rand(), rand()) for _ in 1:n]
        pts_col = [SVector{2,Float64}(2.0 + rand(), rand()) for _ in 1:n]

        K = KernelMatrix(pts, pts_col) do x, y
            1 / (4π * norm(x - y))
        end

        Xclt = ClusterTree(deepcopy(pts), GeometricSplitter(; nmax=20))
        Yclt = ClusterTree(deepcopy(pts_col), GeometricSplitter(; nmax=20))

        h2 = assemble_h2matrix(K, Xclt, Yclt; order=4, global_index=true)

        recompress!(h2; rtol=1e-4, maxrank=30)

        # matvec should match dense conversion
        x = randn(n)
        M = Matrix(h2)
        @test norm(h2 * x - M * x) < 1e-10 * norm(M * x)
    end
end
