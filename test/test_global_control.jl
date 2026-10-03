# Global (absolute) error control: basis threshold, ACA tolerance, operator
# scale estimate, global coupling truncation and factored packet couplings.

# Algebraically decaying 2D kernel: far blocks have norms far below the near
# field, so absolute thresholds truncate them more than block-relative ones.
function decaying_kernel_problem(n; seed=17, nmax=12)
    rng = MersenneTwister(seed)
    X = [Point2D(rand(rng), rand(rng)) for _ in 1:n]
    K = KernelMatrix(X, X) do x, y
        r = norm(x - y)
        1 / (r^3 + 1e-3)
    end
    tree = ClusterTree(copy(X), GeometricSplitter(; nmax))
    return K, tree, X
end

dense_of(K, n) = [K[i, j] for i in 1:n, j in 1:n]
relerr(A, B) = norm(A - B) / norm(B)

@testset "Truncation rank with absolute threshold" begin
    tr = H2Matrices._truncation_rank
    S = [1.0, 1e-2, 1e-4, 1e-6, 1e-8]
    @test tr(S, 1e-3, 10) == tr(S, 1e-3, 10, 0.0) == tr(S, 1e-3, 10, 0.0, 1e-9) == 2
    @test tr(S, 0.0, 10, 1e-5) == 3                 # absolute only
    @test tr(S, 1e-3, 10, 1e-5) == 2                # max(rtol*S[1], atol)
    @test tr(2e-2 .* S, 1e-3, 10) == 2
    @test tr(2e-2 .* S, 1e-3, 10, 1e-3) == 1        # weak block: absolute threshold dominates
    @test tr(S, 0.0, 10, 1e-1) == 1
    @test tr(S, 0.0, 10, 1e-1, 1e-3) == 2           # relative safeguard keeps S[i] > 1e-3*S[1]
    @test tr(S, 0.0, 10, 2.0) == 0                  # everything below the absolute threshold
    @test tr(S, 0.0, 10, 2.0, 1e-5) == 3
    @test tr(S, 0.0, 2, 1e-9) == 2                  # rank cap
    @test tr(zeros(3), 0.0, 3, 1e-9) == 0
    @test tr(Float64[], 0.0, 3, 1e-9) == 0
end

@testset "Absolute basis truncation in the H → H² conversion" begin
    n = 500
    K, tree, X = decaying_kernel_problem(n)
    H = assemble_hmatrix(K, tree, tree; comp=PartialACA(; rtol=1e-12), threads=false, global_index=true)
    M = dense_of(K, n)
    s = estimate_operator_scale(K, tree, tree)
    conv(; kw...) = compress_hmatrix_to_h2(H; maxrank=n, strict=true, threads=false, _print=false, kw...)
    ranksum(h) = sum(cb.k for cb in H2Matrices.nodes(h.row_basis)) + sum(cb.k for cb in H2Matrices.nodes(h.col_basis))
    ref = conv(; rtol=1e-6)
    # atol=0 (and an inactive safeguard) is the block-relative rule.
    x = randn(MersenneTwister(5), n)
    @test conv(; rtol=1e-6, atol=0.0) * x == ref * x
    @test conv(; rtol=1e-6, atol=0.0, safeguard_rtol=1e-3) * x == ref * x
    # Absolute control: fewer basis vectors than block-relative control at a
    # similar operator error, and the error follows the absolute threshold.
    glob = conv(; rtol=0.0, atol=1e-6 * s)
    @test ranksum(glob) < ranksum(ref)
    eref = relerr(Matrix(ref), M); eglob = relerr(Matrix(glob), M)
    @test eglob < 1e-5
    @test storage_bytes(glob) < storage_bytes(ref)
    @test relerr(Matrix(conv(; rtol=0.0, atol=1e-9 * s)), M) < 1e-8
    # Both criteria at once: max(rtol*σ₁, atol) never keeps more than either.
    both = conv(; rtol=1e-6, atol=1e-6 * s)
    @test ranksum(both) <= min(ranksum(ref), ranksum(glob))
    # A huge absolute threshold drops every basis (near field only); the
    # relative safeguard keeps clusters resolved to its level.
    none = conv(; rtol=0.0, atol=1e6 * s)
    @test ranksum(none) == 0
    safe = conv(; rtol=0.0, atol=1e6 * s, safeguard_rtol=1e-6)
    @test relerr(Matrix(safe), M) < 1e-4
    @test ranksum(safe) > 0
    # Threaded and serial conversions agree bitwise with an absolute threshold.
    if Threads.nthreads() > 1
        a = compress_hmatrix_to_h2(H; rtol=0.0, atol=1e-7 * s, maxrank=n, threads=true, _print=false)
        b = compress_hmatrix_to_h2(H; rtol=0.0, atol=1e-7 * s, maxrank=n, threads=false, _print=false)
        @test a * x == b * x && adjoint(a) * x == adjoint(b) * x
    end
    # Rank caps are judged against the absolute threshold.
    @test_throws ArgumentError conv(; rtol=0.0, atol=1e-12 * s, maxrank=2)
    @test conv(; rtol=0.0, atol=1e3 * s, maxrank=2) isa H2Matrix
    @test_throws ArgumentError conv(; rtol=1e-6, atol=-1.0)
    @test_throws ArgumentError conv(; rtol=1e-6, atol=NaN)
    @test_throws ArgumentError conv(; rtol=1e-6, atol=1e-8, safeguard_rtol=0.0)
    @test_throws ArgumentError compress_hmatrix_to_h2(H; rtol=1e-6, atol=1e-8, maxrank=n, _reference=true, _print=false)
end

@testset "Operator scale estimate" begin
    n = 300
    K, tree, X = decaying_kernel_problem(n; seed=3)
    M = dense_of(K, n)
    exact = norm(M) / sqrt(n)
    @test estimate_operator_scale(K, tree, tree; samples=n) ≈ exact rtol=1e-12
    @test estimate_operator_scale(K, tree, tree; samples=10n) ≈ exact rtol=1e-12
    est = estimate_operator_scale(K, tree, tree)
    @test est == estimate_operator_scale(K, tree, tree)         # deterministic
    @test 0.5exact < est < 2exact
    # Local indexing reads K in tree order directly.
    p = HMatrices.loc2glob(tree)
    Kloc = KernelMatrix((x, y) -> 1 / (norm(x - y)^3 + 1e-3), X[p], X[p])
    @test estimate_operator_scale(Kloc, tree, tree; global_index=false, samples=n) ≈ exact rtol=1e-12
    @test_throws ArgumentError estimate_operator_scale(K, tree, tree; samples=0)
end

@testset "error_control=:global in adaptive assembly" begin
    n = 400
    K, tree, X = decaying_kernel_problem(n; seed=11)
    M = dense_of(K, n)
    x = randn(MersenneTwister(2), n)
    kw = (; rtol=1e-7, aca_rtol=1e-8, maxrank=n, strict=true, conversion_threads=false)
    blk = assemble_h2matrix_adaptive(K, tree, tree; kw...)
    @test assemble_h2matrix_adaptive(K, tree, tree; kw..., error_control=:block) * x == blk * x
    s = estimate_operator_scale(K, tree, tree)
    glob = assemble_h2matrix_adaptive(K, tree, tree; kw..., error_control=:global)
    @test assemble_h2matrix_adaptive(K, tree, tree; kw..., error_control=:global, scale=s) * x == glob * x
    # Same as the explicit pipeline: absolute ACA tolerance, absolute basis threshold.
    H = assemble_hmatrix(K, tree, tree; comp=PartialACA(; atol=1e-8 * s, rtol=0.0), threads=false, global_index=true)
    manual = compress_hmatrix_to_h2(H; rtol=0.0, atol=1e-7 * s, maxrank=n, strict=true, threads=false, _print=false)
    @test manual * x == glob * x
    @test relerr(Matrix(glob), M) < 1e-6
    @test storage_bytes(glob) < storage_bytes(blk)
    # Explicit absolute tolerances with block control.
    expl = assemble_h2matrix_adaptive(K, tree, tree; kw..., atol=1e-7 * s, aca_atol=1e-8 * s)
    @test relerr(Matrix(expl), M) < 1e-5
    @test_throws ArgumentError assemble_h2matrix_adaptive(K, tree, tree; kw..., error_control=:other)
    @test_throws ArgumentError assemble_h2matrix_adaptive(K, tree, tree; kw..., scale=1.0)
    @test_throws ArgumentError assemble_h2matrix_adaptive(K, tree, tree; kw..., error_control=:global, scale=0.0)
    @test_throws ArgumentError assemble_h2matrix_adaptive(K, tree, tree; kw..., aca_atol=-1.0)
end

@testset "Global coupling truncation and factored packet couplings" begin
    rng = MersenneTwister(77)
    n = 500
    K, tree, X = decaying_kernel_problem(n; seed=29)
    H = assemble_hmatrix(K, tree, tree; comp=PartialACA(; rtol=1e-12), threads=false, global_index=true)
    C = compress_hmatrix_to_h2(H; rtol=1e-11, maxrank=n, strict=true, threads=false, _print=false)
    M = Matrix(C)
    s = estimate_operator_scale(K, tree, tree)
    @test_throws ArgumentError H2CompactMatvecPlan(C; coupling_scale=:global)
    @test_throws ArgumentError H2CompactMatvecPlan(C; coupling_scale=1.0)
    @test_throws ArgumentError H2CompactMatvecPlan(C; coupling_rtol=1e-8, coupling_scale=:other)
    @test_throws ArgumentError H2CompactMatvecPlan(C; coupling_rtol=1e-8, coupling_scale=-1.0)
    @test_throws ArgumentError H2CompactMatvecPlan(C; coupling_rtol=1e-8, coupling_scale=Inf)
    @test_throws ArgumentError H2CompactMatvecPlan(C; coupling_rtol=1e-8, coupling_scale=true)
    blockplan = H2CompactMatvecPlan(C; coupling_rtol=1e-6)
    @test all(((a, b),) -> a.L == b.L && a.R == b.R,
              zip(blockplan.couplings, H2CompactMatvecPlan(C; coupling_rtol=1e-6, coupling_scale=:block).couplings))
    # :global is the largest stored block norm, passed explicitly.
    exact = H2CompactMatvecPlan(C)
    gmax = max(maximum(b -> opnorm(b.S), exact.couplings), maximum(b -> opnorm(b.D), exact.dense))
    g = H2CompactMatvecPlan(C; coupling_rtol=1e-6, coupling_scale=:global)
    @test all(((a, b),) -> a.L == b.L && a.R == b.R,
              zip(g.couplings, H2CompactMatvecPlan(C; coupling_rtol=1e-6, coupling_scale=gmax).couplings))
    # Absolute truncation: each coupling's error is at most τ·scale and
    # storage falls below block-relative truncation at the same τ; with a
    # larger τ, weak couplings are dropped.
    tau = 1e-6
    a = H2CompactMatvecPlan(C; coupling_rtol=tau, coupling_scale=s)
    @test any(b -> b.R !== nothing && size(b.R, 2) > 0, a.couplings)
    @test storage_bytes(a) < storage_bytes(blockplan)
    Ma = reduce(hcat, [a * e for e in eachcol(Matrix(1.0I, n, n))])
    @test relerr(Ma, M) < 1e-5
    a2 = H2CompactMatvecPlan(C; coupling_rtol=3e-2, coupling_scale=s)
    @test any(b -> b.R !== nothing && size(b.R, 2) == 0, a2.couplings)
    # Packet plans keep the factors: same numbers, same products up to rounding.
    for workers in (1, 4), c in (a, a2)
        P = H2PacketMatvecPlan(c; workers)
        @test storage_bytes(P) == storage_bytes(c)
        @test any(b -> !isempty(b.factors), P.packets)
        @test sum(b -> length(b.columns), P.packets) == count(b -> b.R === nothing || size(b.R, 2) > 0, c.couplings)
        x = randn(rng, n); z = randn(rng, n)
        for (A, B, input) in ((P, c, x), (transpose(P), transpose(c), z), (adjoint(P), adjoint(c), z))
            @test A * input ≈ B * input rtol=1e-13
            y = randn(rng, n); old = copy(y); expected = B * input
            mul!(y, A, input, 1.7, -0.3)
            @test y ≈ 1.7expected - 0.3old rtol=1e-13
            fill!(y, NaN); mul!(y, A, input, 0.0, 0.0); @test all(iszero, y)
        end
        @test dot(z, P * x) ≈ dot(adjoint(P) * z, x) rtol=1e-13
        Xk = randn(rng, n, 9); Zk = randn(rng, n, 9)
        Y = P * Xk; W = adjoint(P) * Zk
        @test Y ≈ reduce(hcat, [P * c for c in eachcol(Xk)]) rtol=1e-14
        @test W ≈ reduce(hcat, [adjoint(P) * c for c in eachcol(Zk)]) rtol=1e-14
        Q = copy(P)
        @test all(b.factors === c.factors for (b, c) in zip(P.packets, Q.packets))
        @test Q * x == P * x
        # Multiplying the factors out stores the dense couplings instead.
        Pm = H2PacketMatvecPlan(c; workers, keep_factors=false)
        @test all(b -> isempty(b.factors), Pm.packets)
        @test storage_bytes(Pm) > storage_bytes(P)
        @test Pm * x ≈ P * x rtol=1e-13
    end
    # Bitwise independent of the worker count, single and several vectors.
    P1 = H2PacketMatvecPlan(a; workers=1); x = randn(rng, n); z = randn(rng, n); Xk = randn(rng, n, 5)
    for w in (2, 3, 7)
        Pw = H2PacketMatvecPlan(a; workers=w)
        @test Pw * x == P1 * x && adjoint(Pw) * z == adjoint(P1) * z
        @test Pw * Xk == P1 * Xk && adjoint(Pw) * Xk == adjoint(P1) * Xk
    end
    # Releasing compact blocks while packing gives the same plan.
    rel = H2PacketMatvecPlan(H2CompactMatvecPlan(C; coupling_rtol=tau, coupling_scale=s); workers=2, _release=true)
    @test rel * x == H2PacketMatvecPlan(a; workers=2) * x
    @test adjoint(rel) * z == adjoint(H2PacketMatvecPlan(a; workers=2)) * z
    # Dropping every coupling leaves the near field only.
    d = H2CompactMatvecPlan(C; coupling_rtol=1.0, coupling_scale=1e6 * s)
    @test all(b -> size(b.R, 2) == 0, d.couplings)
    Pd = H2PacketMatvecPlan(d; workers=2)
    @test isempty(Pd.packets)
    near = zeros(n, n)
    for b in exact.dense
        near[exact.rowperm[b.rows], exact.colperm[b.cols]] .+= b.D
    end
    @test Pd * x ≈ near * x rtol=1e-13
    @test adjoint(Pd) * z ≈ near' * z rtol=1e-13
    function allocations(A, x, y)
        mul!(y, A, x)
        @allocated mul!(y, A, x)
    end
    if VERSION >= v"1.10"
        @test allocations(P1, x, zeros(n)) == 0
        @test allocations(adjoint(P1), z, zeros(n)) == 0
        @test allocations(P1, randn(rng, n, 5), zeros(n, 5)) == 0
        @test allocations(adjoint(P1), randn(rng, n, 3), zeros(n, 3)) == 0
    end
end
