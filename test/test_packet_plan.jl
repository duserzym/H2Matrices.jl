@testset "Packed coupling products and race-free adjoints" begin
    rng=MersenneTwister(920)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:235]
    Y=[Point2D(rand(rng),rand(rng)) for _ in 1:171]
    K=KernelMatrix(X,Y) do x,y
        exp(-sum(abs2,x-y))*(1+x[1]-0.3y[2])*cos(16*(x[1]*y[2]+x[2]*y[1]))
    end
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=8));ct=ClusterTree(copy(Y),GeometricSplitter(;nmax=8))
    H=assemble_hmatrix(K,rt,ct;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=235,strict=true,_print=false)
    for gi in (true,false),workers in (1,4),crtol in (nothing,1e-12),pt in (false,true),prec in (Float64,Float32,Float16)
        prec!==Float64 && crtol===nothing && continue
        C.global_index=gi;compact=H2CompactMatvecPlan(C;coupling_rtol=crtol,passthrough=pt,coupling_precision=prec)
        P=H2PacketMatvecPlan(compact;workers);M=Matrix(C;global_index=gi)
        @test !(:h2 in fieldnames(typeof(P)))
        x=randn(rng,length(Y));z=randn(rng,length(X))
        for (A,input,expected) in ((P,x,M*x),(transpose(P),z,M'*z),(adjoint(P),z,M'*z))
            @test A*input ≈ expected rtol=1e-10 atol=1e-11
            y=randn(rng,length(expected));old=copy(y)
            mul!(y,A,input,1.7,-0.3)
            @test y ≈ 1.7expected-0.3old rtol=1e-10 atol=1e-11
            fill!(y,NaN);mul!(y,A,input,0.,0.);@test all(iszero,y)
            @test_throws DimensionMismatch mul!(zeros(1),A,input)
        end
        @test dot(z,P*x) ≈ dot(adjoint(P)*z,x) rtol=1e-12 atol=1e-12
        Q=copy(P)
        @test Q.rowcoeff!==P.rowcoeff && Q.colcoeff!==P.colcoeff
        @test all(b.matrix===c.matrix && b.scratch!==c.scratch for (b,c) in zip(P.packets,Q.packets))
        # Includes nested task scheduling when each independent caller has workers.
        f=Threads.@spawn P*x
        g=Threads.@spawn adjoint(Q)*z
        @test fetch(f) ≈ M*x rtol=1e-10 atol=1e-11
        @test fetch(g) ≈ M'*z rtol=1e-10 atol=1e-11
    end
    # Artificial overlapping near-field row ranges exercise safe forward
    # fallback and private transpose reduction, beyond geometric leaf layouts.
    C.global_index=true;core=H2CompactMatvecPlan(C)
    D=fill(0.03,20,10)
    dense=vcat(core.dense,[H2Matrices._PlanDense(D,1:20,1:10)])
    overlap=H2CompactMatvecPlan(core.shape,core.rows,core.cols,core.couplings,dense,core.rowcoeff,core.colcoeff,core.rowbuffer,core.colbuffer,core.rowperm,core.colperm)
    M=Matrix(C);M[core.rowperm[1:20],core.colperm[1:10]].+=D
    P=H2PacketMatvecPlan(overlap;workers=4)
    @test !P.near_parallel
    x=randn(rng,171);z=randn(rng,235)
    @test P*x ≈ M*x rtol=1e-12 atol=1e-12
    @test adjoint(P)*z ≈ M'*z rtol=1e-12 atol=1e-12
    @test_throws ArgumentError H2PacketMatvecPlan(C;workers=0)
    C.global_index=true;P=H2PacketMatvecPlan(C);x=randn(rng,171);y=zeros(235)
    function allocations(A,x,y)
        mul!(y,A,x)
        @allocated mul!(y,A,x)
    end
    if VERSION>=v"1.10"
        @test allocations(P,x,y)==0
        @test allocations(adjoint(P),randn(rng,235),zeros(171))==0
    end
end
