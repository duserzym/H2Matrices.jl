# Options of different stages used together: global error control in the
# assembly, pass-through bases, global/numeric coupling scales, factored and
# reduced-precision couplings in packets, and the mixed-precision packet plan.

function composed_ring_problem(n; nmax=8, rtol=1e-6, seed=17)
    rng = MersenneTwister(seed)
    pts = [SVector(cos(2π * t) * (1 + 0.2rand(rng)), sin(2π * t) * (1 + 0.2rand(rng)), rand(rng)) for t in rand(rng, n)]
    K = KernelMatrix(pts, pts) do x, y
        r = norm(x - y)
        r == 0 ? 0.0 : inv(r)
    end
    tree = ClusterTree(copy(pts), GeometricSplitter(; nmax))
    return K, tree
end

@testset "Mixed-precision packets on pass-through bases" begin
    rng = MersenneTwister(5)
    n = 1500
    K, tree = composed_ring_problem(n)
    H = assemble_hmatrix(K, tree, tree; comp=PartialACA(; rtol=1e-7), threads=false, global_index=true)
    C = compress_hmatrix_to_h2(H; rtol=1e-6, maxrank=n, strict=true, threads=false, _print=false)
    M = Matrix(C); x = randn(rng, n); z = randn(rng, n); X = randn(rng, n, 7); Z = randn(rng, n, 7)
    F = H2CompactMatvecPlan(C; passthrough=true)
    # Children of pass-through nodes share their coefficients with the parent,
    # so their packet rotations must be stored, not absorbed into the basis.
    shared = [nd.identity || nd.passthrough for nd in F.rows]
    for nd in F.rows
        nd.passthrough && (shared[nd.children] .= true)
    end
    children = Set(i for i in eachindex(F.rows) if shared[i] && !F.rows[i].identity && !F.rows[i].passthrough && !isempty(F.rows[i].coeff))
    @test !isempty(children) && any(nd -> nd.passthrough, F.rows)
    rowof = Dict(first(nd.coeff) => i for (i, nd) in enumerate(F.rows) if !isempty(nd.coeff))
    P0 = H2PacketMatvecPlan(F; workers=2)
    for rtol in (0.0, 1e-12, 1e-9)
        P1 = H2MixedPacketMatvecPlan(F; workers=1, precision_rtol=rtol)
        P4 = H2MixedPacketMatvecPlan(F; workers=4, precision_rtol=rtol)
        s = precision_summary(P4)
        @test s.bound <= rtol * s.reference_norm * (1 + 1e-12)
        @test s.reference_norm ≈ norm(M) rtol=1e-10
        # Rotated packets of pass-through children keep explicit reflectors.
        rtol == 1e-9 && @test any(b -> rowof[first(b.row)] in children && !isempty(b.tau), P4.engine.packets)
        @test norm(P4 * x - M * x) <= s.bound * norm(x) + 1e-13 * opnorm(M) * norm(x)
        @test norm(adjoint(P4) * z - M' * z) <= s.bound * norm(z) + 1e-13 * opnorm(M) * norm(z)
        @test P4 * x == P1 * x && adjoint(P4) * z == adjoint(P1) * z
        @test P4 * X == P1 * X && adjoint(P4) * Z == adjoint(P1) * Z
        Y = P4 * X
        for v in axes(X, 2)
            @test Y[:, v] ≈ P4 * X[:, v] rtol=1e-13
        end
        @test dot(z, P4 * x) ≈ dot(adjoint(P4) * z, x) rtol=1e-13
        if rtol == 0
            @test P4 * x == P0 * x && adjoint(P4) * z == adjoint(P0) * z && P4 * X == P0 * X
            @test storage_bytes(P4) == storage_bytes(P0)
        else
            @test storage_bytes(P4) < storage_bytes(P0)
        end
    end
    # Built from the H² matrix with compact options; consuming gives the same plan.
    ref = H2MixedPacketMatvecPlan(F; workers=2, precision_rtol=1e-11)
    D = deepcopy(C)
    Pc = H2MixedPacketMatvecPlan(D; workers=2, precision_rtol=1e-11, passthrough=true, consume=true)
    @test Pc * x == ref * x && adjoint(Pc) * z == adjoint(ref) * z
    @test storage_bytes(Pc) == storage_bytes(ref)
    @test all(l -> l.uniform === nothing && l.dense === nothing, H2Matrices.leaves(D))
    # Factorized couplings are multiplied out; the bound is relative to the
    # truncated compact operator.
    Ft = H2CompactMatvecPlan(C; passthrough=true, coupling_rtol=1e-7, coupling_scale=:global)
    @test any(b -> b.R !== nothing, Ft.couplings)
    Pt = H2MixedPacketMatvecPlan(Ft; workers=3, precision_rtol=1e-10); st = precision_summary(Pt)
    @test norm(Pt * x - Ft * x) <= st.bound * norm(x) + 1e-13 * opnorm(M) * norm(x)
    # Reduced-precision couplings are rejected before anything is consumed.
    D = deepcopy(C)
    @test_throws ArgumentError H2MixedPacketMatvecPlan(D; consume=true, coupling_rtol=1e-8, coupling_precision=Float32)
    @test D * x == C * x
    @test_throws ArgumentError H2MixedPacketMatvecPlan(H2CompactMatvecPlan(C; coupling_rtol=1e-8, coupling_precision=Float16))
    @test_throws ArgumentError H2MixedPacketMatvecPlan(deepcopy(C); consume=true, precision_rtol=-1.0)
    @test_throws ArgumentError H2MixedPacketMatvecPlan(deepcopy(C); workers=0)
    function allocations(A, x, y)
        mul!(y, A, x)
        @allocated mul!(y, A, x)
    end
    if VERSION >= v"1.10"
        P1 = H2MixedPacketMatvecPlan(F; workers=1, precision_rtol=1e-9)
        @test allocations(P1, x, zeros(n)) == 0
        @test allocations(adjoint(P1), z, zeros(n)) == 0
        @test allocations(P1, X, zeros(n, 7)) == 0
    end
end

@testset "Global error control with pass-through, scaled and reduced-precision couplings" begin
    rng = MersenneTwister(9)
    n = 900
    K, tree = composed_ring_problem(n; nmax=16, seed=23)
    Md = [K[i, j] for i in 1:n, j in 1:n]
    s = estimate_operator_scale(K, tree, tree)
    kw = (; rtol=1e-8, aca_rtol=1e-9, maxrank=n, strict=true, conversion_threads=false)
    h2 = assemble_h2matrix_adaptive(K, tree, tree; kw..., error_control=:global, scale=s)
    x = randn(rng, n); z = randn(rng, n); X = randn(rng, n, 5)
    @test norm(h2 * x - Md * x) / norm(Md * x) < 1e-6
    for opts in ((; passthrough=true, coupling_rtol=1e-8, coupling_scale=s),
                 (; passthrough=true, coupling_rtol=1e-8, coupling_scale=s, coupling_precision=Float16),
                 (; passthrough=true, coupling_rtol=1e-8, coupling_scale=:global, coupling_precision=Float32))
        F = H2CompactMatvecPlan(h2; opts...)
        P = H2PacketMatvecPlan(F; workers=4)
        @test storage_bytes(P) == storage_bytes(F) < storage_bytes(H2CompactMatvecPlan(h2))
        @test norm(P * x - Md * x) / norm(Md * x) < 1e-6
        @test norm(adjoint(P) * z - Md' * z) / norm(Md' * z) < 1e-6
        @test P * x ≈ F * x rtol=1e-13
        @test P * X == H2PacketMatvecPlan(F; workers=1) * X
        # The consuming one-step build equals the compact-plan route bitwise.
        D = deepcopy(h2)
        Pc = H2PacketMatvecPlan(D; workers=4, consume=true, opts...)
        @test Pc * x == P * x && adjoint(Pc) * z == adjoint(P) * z && Pc * X == P * X
        @test storage_bytes(Pc) == storage_bytes(P)
    end
    # Numeric and :global coupling scales agree when given the same value.
    exact = H2CompactMatvecPlan(h2)
    gmax = max(maximum(b -> opnorm(b.S), exact.couplings), maximum(b -> opnorm(b.D), exact.dense))
    a = H2PacketMatvecPlan(H2CompactMatvecPlan(h2; coupling_rtol=1e-8, coupling_scale=:global, coupling_precision=Float16))
    b = H2PacketMatvecPlan(H2CompactMatvecPlan(h2; coupling_rtol=1e-8, coupling_scale=gmax, coupling_precision=Float16))
    @test a * x == b * x && storage_bytes(a) == storage_bytes(b)
    # The mixed-precision plan composes with globally controlled bases.
    Pm = H2MixedPacketMatvecPlan(h2; workers=2, precision_rtol=1e-12, passthrough=true)
    sm = precision_summary(Pm)
    Fm = H2CompactMatvecPlan(h2; passthrough=true)
    @test norm(Pm * x - Fm * x) <= sm.bound * norm(x) + 1e-13 * opnorm(Md) * norm(x)
    @test storage_bytes(Pm) < storage_bytes(H2PacketMatvecPlan(Fm))
end
