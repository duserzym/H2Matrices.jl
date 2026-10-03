function _same_h2_data(a::H2Matrix, b::H2Matrix)
    la = H2Matrices.leaves(a); lb = H2Matrices.leaves(b)
    length(la) == length(lb) || return false
    for (x, y) in zip(la, lb)
        (x.uniform === nothing) == (y.uniform === nothing) || return false
        x.uniform === nothing || x.uniform.S == y.uniform.S || return false
        x.dense == y.dense || return false
    end
    for (ra, rb) in ((a.row_basis, b.row_basis), (a.col_basis, b.col_basis))
        for (x, y) in zip(H2Matrices.nodes(ra), H2Matrices.nodes(rb))
            x.k == y.k && x.V == y.V && x.E == y.E || return false
        end
    end
    return true
end

_node_ranks(h) = ([b.k for b in H2Matrices.nodes(h.row_basis)], [b.k for b in H2Matrices.nodes(h.col_basis)])

function _all_leaf_data_released(H)
    if HMatrices.isleaf(H)
        return HMatrices.data(H) === nothing
    end
    return all(_all_leaf_data_released, HMatrices.children(H))
end

@testset "Condensed H → H² conversion" begin
    @testset "Exact versus reference construction (sphere, saturation and truncation)" begin
        rng = MersenneTwister(931)
        X = [Point3D(normalize(randn(rng, 3))...) for _ in 1:900]
        K = KernelMatrix(X, X) do x, y
            r = norm(x - y)
            r > 0 ? 1 / (4π * r) : 0.0
        end
        rt = ClusterTree(copy(X), GeometricSplitter(; nmax=16))
        ct = ClusterTree(copy(X), GeometricSplitter(; nmax=16))
        H = assemble_hmatrix(K, rt, ct; comp=PartialACA(; rtol=1e-12), global_index=true, threads=false)
        ref = compress_hmatrix_to_h2(H; rtol=1e-10, maxrank=400, strict=true, _reference=true, _print=false)
        new = compress_hmatrix_to_h2(H; rtol=1e-10, maxrank=400, strict=true, threads=false, _print=false)
        thr = compress_hmatrix_to_h2(H; rtol=1e-10, maxrank=400, strict=true, threads=true, _print=false)
        # Condensation leaves every truncation decision unchanged.
        @test _node_ranks(new) == _node_ranks(ref)
        Mref = Matrix(ref); Mnew = Matrix(new)
        @test norm(Mnew - Mref) / norm(Mref) < 1e-13
        # Deterministic: bitwise independent of the thread count.
        @test _same_h2_data(new, thr)
        # The test exercises identity leaves, identity embeddings and truncated bases.
        nodes = H2Matrices.nodes(new.row_basis)
        @test any(b -> H2Matrices.isleaf(b) && b.k == length(b) && b.V == I, nodes)
        @test any(b -> !H2Matrices.isleaf(b) && b.k > 0 && b.k == length(b), nodes)
        @test any(b -> 0 < b.k < length(b), nodes)
        # Compact plans skip exact identity expansions (bitwise equivalent) and
        # agree with the reference operator.
        rng2 = MersenneTwister(7); x = randn(rng2, 900); z = randn(rng2, 900)
        Pn = H2CompactMatvecPlan(new); Pr = H2CompactMatvecPlan(ref)
        @test norm(Pn*x - Pr*x) / norm(Pr*x) < 1e-13
        @test norm(adjoint(Pn)*z - adjoint(Pr)*z) / norm(adjoint(Pr)*z) < 1e-13
        @test Pn*x ≈ Matrix(new; global_index=true)*x rtol=1e-12
        @test dot(z, Pn*x) ≈ dot(adjoint(Pn)*z, x) rtol=1e-13
    end

    @testset "Rectangular trees and rescaled ACA factors" begin
        rng = MersenneTwister(932)
        X = [Point2D(rand(rng), rand(rng)) for _ in 1:235]
        Y = [Point2D(0.2 + rand(rng), rand(rng)) for _ in 1:171]
        K = KernelMatrix(X, Y) do x, y
            exp(-sum(abs2, x-y)) * (1 + x[1] - 0.2y[2])
        end
        rt = ClusterTree(copy(X), GeometricSplitter(; nmax=8))
        ct = ClusterTree(copy(Y), GeometricSplitter(; nmax=8))
        H = assemble_hmatrix(K, rt, ct; comp=PartialACA(; rtol=1e-12), global_index=true, threads=false)
        for rtol in (1e-10, 1e-6)
            ref = compress_hmatrix_to_h2(H; rtol, maxrank=120, strict=true, _reference=true, _print=false)
            new = compress_hmatrix_to_h2(H; rtol, maxrank=120, strict=true, _print=false)
            @test _node_ranks(new) == _node_ranks(ref)
            @test norm(Matrix(new) - Matrix(ref)) / norm(Matrix(ref)) < 1e-12
        end
    end

    @testset "Consuming conversion and packet construction" begin
        rng = MersenneTwister(933)
        X = [Point3D(normalize(randn(rng, 3))...) for _ in 1:600]
        K = KernelMatrix(X, X) do x, y
            r = norm(x - y)
            r > 0 ? (1 + 0.1x[1]) / r : 0.0
        end
        rt = ClusterTree(copy(X), GeometricSplitter(; nmax=16))
        ct = ClusterTree(copy(X), GeometricSplitter(; nmax=16))
        H = assemble_hmatrix(K, rt, ct; comp=PartialACA(; rtol=1e-11), global_index=true, threads=false)
        kept = compress_hmatrix_to_h2(H; rtol=1e-9, maxrank=300, strict=true, _print=false)
        # Without consumption a strict rank-cap failure leaves H usable for a retry;
        # with consumption the failure is still reported (H is then unusable).
        Hs = deepcopy(H)
        @test_throws ArgumentError compress_hmatrix_to_h2(Hs; rtol=1e-12, maxrank=1, strict=true, _print=false)
        @test !_all_leaf_data_released(Hs)
        @test _same_h2_data(compress_hmatrix_to_h2(Hs; rtol=1e-9, maxrank=300, strict=true, _print=false), kept)
        @test_throws ArgumentError compress_hmatrix_to_h2(deepcopy(H); rtol=1e-12, maxrank=1, strict=true,
                                                          consume=true, _print=false)
        for threads in (false, true)
            Hc = deepcopy(H)
            used = compress_hmatrix_to_h2(Hc; rtol=1e-9, maxrank=300, strict=true, consume=true,
                                          threads, _print=false)
            @test _same_h2_data(used, kept)
            @test _all_leaf_data_released(Hc)
        end

        x = randn(rng, 600); z = randn(rng, 600)
        for workers in (1, 4)
            P = H2PacketMatvecPlan(kept; workers)
            source = copy(kept)
            Q = H2PacketMatvecPlan(source; workers, consume=true)
            @test all(l -> l.uniform === nothing && l.dense === nothing, H2Matrices.leaves(source))
            @test all(p.matrix == q.matrix for (p, q) in zip(P.packets, Q.packets))
            @test all(p.matrix == q.matrix for (p, q) in zip(P.nearpackets, Q.nearpackets))
            @test P*x == Q*x
            @test adjoint(P)*z == adjoint(Q)*z
            @test Q*x ≈ Matrix(kept)*x rtol=1e-12
            @test dot(z, Q*x) ≈ dot(adjoint(Q)*z, x) rtol=1e-13
        end
        # Consuming adaptive assembly gives the same operator.
        h2 = assemble_h2matrix_adaptive(K, rt, ct; rtol=1e-9, maxrank=300, aca_rtol=1e-11, strict=true)
        @test _same_h2_data(h2, kept)
    end

    @testset "Rank caps, loose tolerances and near-field-only matrices" begin
        rng = MersenneTwister(935)
        X = [Point3D(normalize(randn(rng, 3))...) for _ in 1:500]
        K = KernelMatrix(X, X) do x, y
            r = norm(x - y)
            r > 0 ? 1 / r : 0.0
        end
        H = assemble_hmatrix(K, ClusterTree(copy(X), GeometricSplitter(; nmax=16)),
            ClusterTree(copy(X), GeometricSplitter(; nmax=16)); comp=PartialACA(; rtol=1e-12),
            global_index=true, threads=false)
        for (rtol, maxrank) in ((1e-10, 12), (1e-4, 400), (0.0, 400))
            ref = compress_hmatrix_to_h2(H; rtol, maxrank, _reference=true, _print=false)
            new = maxrank == 12 ?
                (@test_logs (:warn, r"rank cap") match_mode=:any compress_hmatrix_to_h2(H; rtol, maxrank, _print=false)) :
                compress_hmatrix_to_h2(H; rtol, maxrank, _print=false)
            @test _node_ranks(new) == _node_ranks(ref)
            @test norm(Matrix(new) - Matrix(ref)) / norm(Matrix(ref)) < 1e-13
        end
        # rtol = 0 keeps every direction: identity bases reproduce H exactly.
        @test Matrix(compress_hmatrix_to_h2(H; rtol=0.0, maxrank=400, _print=false)) == Matrix(H)
        Y = X[1:10]
        K2 = KernelMatrix(Y, Y) do x, y
            r = norm(x - y)
            r > 0 ? 1 / r : 0.0
        end
        H2 = assemble_hmatrix(K2, ClusterTree(copy(Y), GeometricSplitter(; nmax=32)),
            ClusterTree(copy(Y), GeometricSplitter(; nmax=32)); global_index=true, threads=false)
        h = compress_hmatrix_to_h2(deepcopy(H2); rtol=1e-10, maxrank=10, consume=true, _print=false)
        @test Matrix(h) == Matrix(H2)
        x = randn(rng, 10)
        @test H2PacketMatvecPlan(h; workers=2, consume=true)*x ≈ Matrix(H2)*x
    end

    @testset "Saturation certificate" begin
        rng = MersenneTwister(934)
        L = Matrix(UpperTriangular(randn(rng, 40, 40))) + 10I
        @test H2Matrices._certified_full_rank(L, 1e-10)
        L[end, end] = 1e-14
        @test !H2Matrices._certified_full_rank(L, 1e-10)
        L[end, end] = 0.0
        @test !H2Matrices._certified_full_rank(L, 0.0)
        @test !H2Matrices._certified_full_rank(zeros(3, 3), 1e-10)
        # Certified matrices are indeed untruncated by the SVD rule.
        G = Matrix(UpperTriangular(randn(rng, 30, 30))) + 6I
        @test H2Matrices._certified_full_rank(G, 1e-10)
        @test H2Matrices._truncation_rank(svdvals(G), 1e-10, 30) == 30
    end
end
