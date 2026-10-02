@testset "Implicit saturated bases preserve stored operators" begin
    rng=MersenneTwister(301)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:235]
    Y=[Point2D(rand(rng),rand(rng)) for _ in 1:171]
    K=KernelMatrix(X,Y) do x,y
        exp(-sum(abs2,x-y))*(1+x[1]-0.3y[2])*cos(16*(x[1]*y[2]+x[2]*y[1]))
    end
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=8));ct=ClusterTree(copy(Y),GeometricSplitter(;nmax=8))
    H=assemble_hmatrix(K,rt,ct;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=235,strict=true,_print=false)
    original=Matrix(C)
    for gi in (true,false), crtol in (nothing,1e-12)
        C.global_index=gi;P=H2CompactMatvecPlan(C;coupling_rtol=crtol)
        M=Matrix(C;global_index=gi)
        @test any(n->n.identity,P.rows)
        @test storage_bytes(P)<=storage_bytes(C)
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
    end
    @test Matrix(C;global_index=true)==original
    @test_throws ArgumentError H2CompactMatvecPlan(C;coupling_rtol=-1.)
    # Interpolation bases are nonorthogonal and can have rank above leaf size.
    K2,r2,c2=separated_laplace2d(80;seed=912)
    A=assemble_h2matrix(K2,r2,c2;order=4,global_index=true)
    P=H2CompactMatvecPlan(A);M=Matrix(A);x=randn(rng,80);z=randn(rng,80)
    @test any(n->n.identity,P.rows)
    @test P*x ≈ M*x rtol=1e-12 atol=1e-12
    @test adjoint(P)*z ≈ M'*z rtol=1e-12 atol=1e-12
    alias=copy(x);mul!(alias,P,alias,1.7,-0.3)
    @test alias ≈ 1.7M*x-0.3x rtol=1e-12 atol=1e-12
    function allocations(A,x,y)
        mul!(y,A,x)
        @allocated mul!(y,A,x)
    end
    if VERSION>=v"1.10"
        @test allocations(P,x,zeros(80))==0
        @test allocations(adjoint(P),x,zeros(80))==0
    end
end
@testset "Independent worker plans share numerical data" begin
    rng=MersenneTwister(821)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:400]
    K=KernelMatrix(X,X) do x,y
        exp(-sum(abs2,x-y))
    end
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=8))
    H=assemble_hmatrix(K,rt,rt;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=400,strict=true,_print=false)
    M=Matrix(C)
    for P in (H2MatvecPlan(C),H2LowRankMatvecPlan(C;rtol=1e-8),H2CompactMatvecPlan(C),H2CompactMatvecPlan(C;coupling_rtol=1e-8))
        Q=copy(P)
        @test Q.rows===P.rows && Q.dense===P.dense
        @test Q.rowcoeff!==P.rowcoeff && Q.colcoeff!==P.colcoeff
        @test Q.rowbuffer!==P.rowbuffer && Q.colbuffer!==P.colbuffer
        @test all(b isa H2Matrices._LowRankPlanCoupling ? (b.L===c.L && b.R===c.R && b.scratch!==c.scratch) : b.S===c.S for (b,c) in zip(P.couplings,Q.couplings))
        inputs=[randn(rng,400) for _ in 1:4]
        workers=[copy(P) for _ in 1:4]
        tasks=[Threads.@spawn begin
            out=zeros(400);back=zeros(400)
            for _ in 1:5
                mul!(out,workers[i],inputs[i]);mul!(back,adjoint(workers[i]),inputs[i])
            end
            (out,back)
        end for i in 1:4]
        for (i,task) in enumerate(tasks)
            out,back=fetch(task)
            @test norm(out-M*inputs[i])/norm(M*inputs[i])<1e-6
            @test norm(back-M'*inputs[i])/norm(M'*inputs[i])<1e-6
        end
    end
end
