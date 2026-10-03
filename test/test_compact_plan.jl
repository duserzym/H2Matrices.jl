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
    for gi in (true,false), crtol in (nothing,1e-12), pt in (false,true), prec in (Float64,Float32,Float16)
        prec!==Float64 && crtol===nothing && continue
        C.global_index=gi;P=H2CompactMatvecPlan(C;coupling_rtol=crtol,passthrough=pt,coupling_precision=prec)
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
@testset "Pass-through bases and factor-keeping packets" begin
    rng=MersenneTwister(7)
    pts=[SVector(cos(2π*t)*(1+0.2rand(rng)),sin(2π*t)*(1+0.2rand(rng)),rand(rng)) for t in rand(rng,1200)]
    K=KernelMatrix(pts,pts) do x,y
        r=norm(x-y);r==0 ? 0.0 : inv(r)
    end
    tree=ClusterTree(copy(pts),GeometricSplitter(;nmax=16))
    H=assemble_hmatrix(K,tree,tree;comp=PartialACA(;rtol=1e-10),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-9,maxrank=1200,strict=true,_print=false)
    M=Matrix(C);x=randn(rng,1200);z=randn(rng,1200)
    base=H2CompactMatvecPlan(C)
    P=H2CompactMatvecPlan(C;passthrough=true)
    @test any(n->n.passthrough,P.rows) && any(n->n.passthrough,P.cols)
    @test !any(n->n.passthrough,base.rows)
    @test storage_bytes(P)<storage_bytes(base)
    for A in (P,H2PacketMatvecPlan(P;workers=1),H2PacketMatvecPlan(P;workers=4))
        for (B,input,expected) in ((A,x,M*x),(transpose(A),z,M'*z),(adjoint(A),z,M'*z))
            @test B*input ≈ expected rtol=1e-12
            y=randn(rng,1200);old=copy(y)
            mul!(y,B,input,1.7,-0.3)
            @test y ≈ 1.7expected-0.3old rtol=1e-12
        end
        @test dot(z,A*x) ≈ dot(adjoint(A)*z,x) rtol=1e-13
        Q=copy(A);@test Q*x ≈ A*x rtol=1e-14
    end
    function allocations(A,x,y)
        mul!(y,A,x)
        @allocated mul!(y,A,x)
    end
    if VERSION>=v"1.10"
        @test allocations(P,x,zeros(1200))==0
        @test allocations(adjoint(P),x,zeros(1200))==0
        @test allocations(H2PacketMatvecPlan(P),x,zeros(1200))==0
        @test allocations(adjoint(H2PacketMatvecPlan(P)),x,zeros(1200))==0
    end
    # Factorized couplings stay factorized inside packets.
    for pt in (false,true),scale in (:block,:global)
        F=H2CompactMatvecPlan(C;coupling_rtol=1e-8,coupling_scale=scale,passthrough=pt)
        @test any(b->b.R!==nothing,F.couplings)
        kept=H2PacketMatvecPlan(F;workers=4);dense=H2PacketMatvecPlan(F;workers=4,keep_factors=false)
        @test any(b->!isempty(b.f64.factors) && !b.plain,kept.packets)
        @test storage_bytes(kept)==storage_bytes(F)
        @test storage_bytes(kept)<storage_bytes(dense)
        for (B,input,ref) in ((kept,x,F*x),(adjoint(kept),z,adjoint(F)*z),(dense,x,F*x))
            @test B*input ≈ ref rtol=1e-13
        end
        @test norm(kept*x-M*x)/norm(M*x)<1e-6
        @test dot(z,kept*x) ≈ dot(adjoint(kept)*z,x) rtol=1e-13
        Q=copy(kept)
        @test Q.packets===kept.packets && all(a!==b for (a,b) in zip(Q.scratch,kept.scratch))
        if VERSION>=v"1.10"
            single=H2PacketMatvecPlan(F)
            @test allocations(single,x,zeros(1200))==0
            @test allocations(adjoint(single),x,zeros(1200))==0
        end
    end
    @test storage_bytes(H2CompactMatvecPlan(C;coupling_rtol=1e-8,coupling_scale=:global))<=
        storage_bytes(H2CompactMatvecPlan(C;coupling_rtol=1e-8))
    @test_throws ArgumentError H2CompactMatvecPlan(C;coupling_rtol=1e-8,coupling_scale=:relative)
    # Mixed precision: Float32 storage for small singular components, Float64 arithmetic.
    @test_throws ArgumentError H2CompactMatvecPlan(C;coupling_precision=Float32)
    @test_throws ArgumentError H2CompactMatvecPlan(C;coupling_rtol=1e-8,coupling_precision=BigFloat)
    @test_throws ArgumentError H2CompactMatvecPlan(C;coupling_precision=Float16)
    # A global scale qualifies coupling truncation too: rejected without it.
    @test_throws ArgumentError H2CompactMatvecPlan(C;coupling_scale=:global)
    @test_throws ArgumentError H2PacketMatvecPlan(C;coupling_scale=:global)
    @test_throws ArgumentError H2PacketMatvecPlan(C;coupling_precision=Float32)
    @test H2CompactMatvecPlan(C;coupling_scale=:block)*x==base*x
    for pt in (false,true),scale in (:block,:global),prec in (Float32,Float16)
        F64=H2CompactMatvecPlan(C;coupling_rtol=1e-9,coupling_scale=scale,passthrough=pt)
        F32=H2CompactMatvecPlan(C;coupling_rtol=1e-9,coupling_scale=scale,passthrough=pt,coupling_precision=prec)
        @test any(b->!isempty(b.L32),F32.couplings)
        @test any(b->b.R32!==nothing,F32.couplings) && any(b->b.R!==nothing && b.R32===nothing && !isempty(b.L32),F32.couplings)
        prec===Float16 && @test any(b->!isempty(b.c16),F32.couplings)
        prec===Float32 && @test all(b->isempty(b.c16),F32.couplings)
        @test storage_bytes(F32)<storage_bytes(F64)
        prec===Float16 && @test storage_bytes(F32)<storage_bytes(H2CompactMatvecPlan(C;coupling_rtol=1e-9,coupling_scale=scale,passthrough=pt,coupling_precision=Float32))
        @test norm(F32*x-M*x)/norm(M*x)<1e-7
        @test norm(adjoint(F32)*z-M'*z)/norm(M'*z)<1e-7
        # Both directions apply the same stored numbers in Float64 arithmetic.
        @test dot(z,F32*x) ≈ dot(adjoint(F32)*z,x) rtol=1e-13
        for workers in (1,4)
            K32=H2PacketMatvecPlan(F32;workers)
            @test storage_bytes(K32)==storage_bytes(F32)
            @test any(b->!isempty(b.f32.columns),K32.packets)
            prec===Float16 && @test any(b->!isempty(b.f16.columns),K32.packets)
            for (B,input,ref) in ((K32,x,F32*x),(transpose(K32),z,transpose(F32)*z),(adjoint(K32),z,adjoint(F32)*z))
                @test B*input ≈ ref rtol=1e-13
                y=randn(rng,1200);old=copy(y)
                mul!(y,B,input,1.7,-0.3)
                @test y ≈ 1.7ref-0.3old rtol=1e-12
            end
            @test dot(z,K32*x) ≈ dot(adjoint(K32)*z,x) rtol=1e-13
            Q=copy(K32)
            @test Q.packets===K32.packets && Q.slots!==K32.slots && all(a!==b for (a,b) in zip(Q.scratch,K32.scratch))
            @test Q*x ≈ K32*x rtol=1e-14
        end
        G=copy(F32);@test G*x ≈ F32*x rtol=1e-14
        @test all(b.L32===c.L32 && (isempty(b.scratch) || b.scratch!==c.scratch) for (b,c) in zip(F32.couplings,G.couplings))
        if VERSION>=v"1.10"
            single=H2PacketMatvecPlan(F32)
            @test allocations(single,x,zeros(1200))==0
            @test allocations(adjoint(single),x,zeros(1200))==0
            @test allocations(F32,x,zeros(1200))==0
            @test allocations(adjoint(F32),x,zeros(1200))==0
        end
    end
    # Mixed-precision kernels match Float64 products of the stored Float32 values.
    A32=randn(rng,Float32,37,23);u=randn(rng,23);v=randn(rng,37)
    @test H2Matrices._mixed_mul!(copy(v),A32,u) ≈ v+Float64.(A32)*u rtol=1e-14
    @test H2Matrices._mixed_tmul!(zeros(23),A32,v,false) ≈ Float64.(A32)'*v rtol=1e-14
    @test H2Matrices._mixed_tmul!(copy(u),A32,v,true) ≈ u+Float64.(A32)'*v rtol=1e-14
    A16=Float16.(A32)
    @test H2Matrices._mixed_mul!(copy(v),A16,u) ≈ v+Float64.(A16)*u rtol=1e-14
    @test H2Matrices._mixed_tmul!(zeros(23),A16,v,false) ≈ Float64.(A16)'*v rtol=1e-14
    @test_throws DimensionMismatch H2Matrices._mixed_mul!(zeros(3),A32,u)
end
@testset "Structured packets in the integrated packet engine" begin
    rng=MersenneTwister(17)
    pts=[SVector(cos(2π*t)*(1+0.2rand(rng)),sin(2π*t)*(1+0.2rand(rng)),rand(rng)) for t in rand(rng,900)]
    K=KernelMatrix(pts,pts) do x,y
        r=norm(x-y);r==0 ? 0.0 : inv(r)
    end
    tree=ClusterTree(copy(pts),GeometricSplitter(;nmax=16))
    H=assemble_hmatrix(K,tree,tree;comp=PartialACA(;rtol=1e-10),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-9,maxrank=900,strict=true,_print=false)
    M=Matrix(C);x=randn(rng,900);z=randn(rng,900);X=randn(rng,900,7);Z=randn(rng,900,7)
    configs=((;passthrough=true),(;passthrough=true,coupling_rtol=1e-9,coupling_scale=:global),
             (;coupling_rtol=1e-9,coupling_precision=Float32),
             (;passthrough=true,coupling_rtol=1e-9,coupling_scale=:global,coupling_precision=Float16))
    for kw in configs
        F=H2CompactMatvecPlan(C;kw...)
        P1=H2PacketMatvecPlan(F;workers=1);P4=H2PacketMatvecPlan(F;workers=4)
        @test storage_bytes(P4)==storage_bytes(F)
        @test !all(b->b.plain,P4.packets) || !haskey(kw,:coupling_rtol)
        # Bitwise independent of the worker count, single and multiple vectors.
        @test P4*x==P1*x && adjoint(P4)*z==adjoint(P1)*z
        @test P4*X==P1*X && adjoint(P4)*Z==adjoint(P1)*Z
        @test P4*x ≈ F*x rtol=1e-13
        @test adjoint(P4)*z ≈ adjoint(F)*z rtol=1e-13
        tol=haskey(kw,:coupling_rtol) ? 1e-7 : 1e-12
        @test norm(P4*x-M*x)/norm(M*x)<tol
        @test dot(z,P4*x) ≈ dot(adjoint(P4)*z,x) rtol=1e-13
        for (A,input) in ((P4,X),(adjoint(P4),Z),(transpose(P1),Z))
            out=A*input
            for v in axes(input,2);@test out[:,v] ≈ A*input[:,v] rtol=1e-13;end
            Y=randn(rng,size(out)...);old=copy(Y)
            mul!(Y,A,input,1.7,-0.3)
            @test Y ≈ 1.7out-0.3old rtol=1e-12
        end
        # Consuming construction from the H2 matrix gives the same plan and
        # leaves the source without numerical blocks.
        D=deepcopy(C)
        Pc=H2PacketMatvecPlan(D;workers=4,consume=true,kw...)
        @test storage_bytes(Pc)==storage_bytes(P4)
        @test Pc*x==P4*x && adjoint(Pc)*z==adjoint(P4)*z && Pc*X==P4*X
        @test all(l->l.uniform===nothing && l.dense===nothing,H2Matrices.leaves(D))
        Fc=H2CompactMatvecPlan(deepcopy(C);consume=true,kw...)
        @test Fc*x==F*x && storage_bytes(Fc)==storage_bytes(F)
        if VERSION>=v"1.10"
            y=zeros(900);mul!(y,P1,x);mul!(y,adjoint(P1),z)
            @test (@allocated mul!(y,P1,x))==0
            @test (@allocated mul!(y,adjoint(P1),z))==0
        end
    end
    # Pass-through without truncation is exact: the plans store the same operator.
    base=H2PacketMatvecPlan(C;workers=2);pt=H2PacketMatvecPlan(C;workers=2,passthrough=true)
    @test storage_bytes(pt)<storage_bytes(base)
    @test pt*X ≈ base*X rtol=1e-13
    @test adjoint(pt)*Z ≈ adjoint(base)*Z rtol=1e-13
    # The Float32 tier is range scaled: tiny or huge operators keep the
    # accuracy of a unit-scale operator, and unit-scale operators store the
    # Float32 factors unscaled.
    F1=H2CompactMatvecPlan(C;coupling_rtol=1e-9,coupling_precision=Float32)
    @test all(b->isempty(b.c32),F1.couplings)
    e1=norm(F1*x-M*x)/norm(M*x)
    for s in (1e-40,1e-200,1e60)
        Cs=deepcopy(C)
        for l in H2Matrices.leaves(Cs)
            l.uniform===nothing || (l.uniform.S.*=s)
            l.dense===nothing || (l.dense.*=s)
        end
        Ms=Matrix(Cs)
        for prec in (Float32,Float16)
            Fs=H2CompactMatvecPlan(Cs;coupling_rtol=1e-9,coupling_precision=prec)
            prec===Float32 && @test any(b->!isempty(b.c32),Fs.couplings)
            Ps=H2PacketMatvecPlan(Fs;workers=2)
            for A in (Fs,Ps)
                @test norm(A*x-Ms*x)/norm(Ms*x)<10*e1
                @test norm(adjoint(A)*z-Ms'*z)/norm(Ms'*z)<1e-8
            end
            @test Ps*x ≈ Fs*x rtol=1e-13
            # Scaled factors in products with several right-hand sides.
            Y=Ps*X;W=adjoint(Ps)*Z
            for v in axes(X,2)
                @test Y[:,v] ≈ Ps*X[:,v] rtol=1e-13
                @test W[:,v] ≈ adjoint(Ps)*Z[:,v] rtol=1e-13
            end
        end
    end
end

