@testset "Nested conversion accuracy and exact stored adjoint" begin
    rng = MersenneTwister(901)
    X = [Point2D(rand(rng), rand(rng)) for _ in 1:235]
    Y = [Point2D(0.2 + rand(rng), rand(rng)) for _ in 1:171]
    K = KernelMatrix(X, Y) do x, y
        exp(-sum(abs2, x-y)) * (1 + x[1] - 0.2y[2])
    end
    rt = ClusterTree(copy(X), GeometricSplitter(; nmax=8))
    ct = ClusterTree(copy(Y), GeometricSplitter(; nmax=8))
    H = assemble_hmatrix(K, rt, ct; comp=PartialACA(; rtol=1e-12),
        global_index=true, threads=false)
    C = compress_hmatrix_to_h2(H; rtol=1e-10, maxrank=120, strict=true, _print=false)
    M = Matrix(H)
    @test norm(Matrix(C)-M)/norm(M) < 2e-9
    @test C.row_basis.k == 0
    @test C.col_basis.k == 0
    @test maximum(b.k for basis in (C.row_basis,C.col_basis) for b in H2Matrices.nodes(basis)) <= 120
    x = randn(rng, length(Y)); z = randn(rng, length(X))
    for A in (C, copy(C))
        A === C || recompress!(A; rtol=1e-10, maxrank=120)
        D = Matrix(A)
        for At in (transpose(A), adjoint(A))
            @test parent(At) === A
            @test At*z ≈ D'*z rtol=1e-12 atol=1e-12
            @test dot(z, A*x) ≈ dot(At*z, x) rtol=1e-12 atol=1e-12
            y = randn(rng, length(Y)); old = copy(y)
            mul!(y, At, z, 1.7, -0.3)
            @test y ≈ 1.7D'*z - 0.3old rtol=1e-12 atol=1e-12
            fill!(y, NaN); mul!(y, At, z, 0.0, 0.0)
            @test all(iszero, y)
            @test_throws DimensionMismatch mul!(zeros(length(Y)-1), At, z)
        end
        A.global_index = false
        local_matrix = Matrix(A; global_index=false)
        @test adjoint(A)*z ≈ local_matrix'*z rtol=1e-12 atol=1e-12
        A.global_index = true
    end

    # Rescale each ACA rank-one term while preserving every represented block.
    function rescale_factors!(H)
        if HMatrices.isleaf(H)
            if HMatrices.isadmissible(H) && HMatrices.data(H) !== nothing
                d = HMatrices.data(H)
                for j in axes(d.A,2)
                    scale = 10.0^(isodd(j) ? 6 : -6)
                    d.A[:,j] .*= scale; d.B[:,j] ./= scale
                end
            end
        else
            foreach(rescale_factors!, HMatrices.children(H))
        end
    end
    rescale_factors!(H)
    scaled = compress_hmatrix_to_h2(H; rtol=1e-10, maxrank=120, strict=true, _print=false)
    @test norm(Matrix(scaled)-M)/norm(M) < 2e-9
    @test norm(Matrix(scaled)-Matrix(C))/norm(M) < 2e-9
    @test_throws ArgumentError compress_hmatrix_to_h2(H; rtol=1e-12, maxrank=1, strict=true, _print=false)
    @test_throws ArgumentError compress_hmatrix_to_h2(H; rtol=NaN, _print=false)
    @test_throws ArgumentError compress_hmatrix_to_h2(H; maxrank=0, _print=false)
end

@testset "Inherited interactions survive internal basis truncation" begin
    # Synthetic mixed-level factor data isolates an ancestor direction that is
    # absent from a node's own interactions. No kernel/ACA error obscures it.
    X = [Point2D(i, 0.01i) for i in 1:64]
    tree = ClusterTree(copy(X), GeometricSplitter(; nmax=8))
    cb = H2Matrices.build_cluster_basis(tree)
    child = cb.children[1]
    m = length(child)
    inherited = ones(length(cb), 1)
    direct = reshape([isodd(i) ? 1.0 : -1.0 for i in 1:m], m, 1)
    data = Dict(objectid(child.cluster) => [(A=direct, B=ones(1,1))])
    H2Matrices._build_adaptive_basis_recursive!(cb, data,
        [(inherited, index_range(tree))]; rtol=1e-12, maxrank=64)
    V = H2Matrices._full_basis(child)
    @test norm(ones(m)-V*(V'*ones(m))) < 1e-10
    @test norm(direct-V*(V'*direct)) < 1e-10
end
