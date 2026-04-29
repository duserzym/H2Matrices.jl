@testset "Cluster bases and Chebyshev interpolation" begin
    pts = random_points2d(160; seed=1)
    clt = ClusterTree(deepcopy(pts), GeometricSplitter(; nmax=20))

    cb = H2Matrices.build_cluster_basis(clt)
    @test cb.cluster === clt
    @test H2Matrices.isroot(cb)
    @test length(H2Matrices.leaves(cb)) > 1

    nodes = H2Matrices.chebyshev_nodes(5)
    @test length(nodes) == 5
    @test all(-1 .<= nodes .<= 1)

    scaled = H2Matrices.chebyshev_nodes_scaled(5, 0.0, 1.0)
    @test all(0 .<= scaled .<= 1)

    bbox = HMatrices.HyperRectangle(SVector(0.0, 0.0), SVector(1.0, 1.0))
    @test length(H2Matrices.chebyshev_interpolation_points(3, bbox)) == 9

    bbox3 = HMatrices.HyperRectangle(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0))
    @test length(H2Matrices.chebyshev_interpolation_points(3, bbox3)) == 27

    H2Matrices.build_chebyshev_basis!(cb, 3)
    for leaf in H2Matrices.leaves(cb)
        @test size(leaf.V, 1) == length(index_range(leaf.cluster))
        @test size(leaf.V, 2) == 9
        @test leaf.k == 9
    end

    cb_copy = copy(cb)
    @test cb_copy !== cb
    @test H2Matrices.total_rank(cb_copy) == H2Matrices.total_rank(cb)
    @test length(H2Matrices.nodes(cb_copy)) == length(H2Matrices.nodes(cb))
end
