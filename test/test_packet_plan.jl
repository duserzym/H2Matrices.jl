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
    for gi in (true,false),workers in (1,4),crtol in (nothing,1e-12)
        C.global_index=gi;compact=H2CompactMatvecPlan(C;coupling_rtol=crtol)
        P=H2PacketMatvecPlan(compact;workers);M=Matrix(C;global_index=gi)
        @test !(:h2 in fieldnames(typeof(P)))
        crtol===nothing && @test storage_bytes(P)==storage_bytes(compact)
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
        @test Q.rowcoeff!==P.rowcoeff && Q.colcoeff!==P.colcoeff && Q.slots!==P.slots
        @test all(a!==b for (a,b) in zip(Q.scratch,P.scratch))
        @test all(b.matrix===c.matrix for (b,c) in zip(P.packets,Q.packets))
        @test all(b.matrix===c.matrix for (b,c) in zip(P.nearpackets,Q.nearpackets))
        # Includes nested task scheduling when each independent caller has workers.
        f=Threads.@spawn P*x
        g=Threads.@spawn adjoint(Q)*z
        @test fetch(f) ≈ M*x rtol=1e-10 atol=1e-11
        @test fetch(g) ≈ M'*z rtol=1e-10 atol=1e-11
    end
    # Every phase has explicit write ownership and a fixed evaluation order,
    # so products are bitwise independent of the worker count.
    C.global_index=true;core=H2CompactMatvecPlan(C)
    P1=H2PacketMatvecPlan(core;workers=1);x=randn(rng,171);z=randn(rng,235)
    for w in (2,3,7)
        Pw=H2PacketMatvecPlan(core;workers=w)
        @test Pw*x==P1*x
        @test adjoint(Pw)*z==adjoint(P1)*z
    end
    # Near-field rows of different tree levels overlap; packets own disjoint
    # elementary intervals after splitting.
    rows=sort([b.rows for b in core.dense];by=first)
    @test any(last(rows[i])>=first(rows[i+1]) for i in 1:length(rows)-1)
    # Artificial overlapping, non-cluster near-field ranges exercise splitting
    # beyond geometric leaf layouts, in both product directions.
    D=fill(0.03,20,10);D2=reshape(collect(1.:21),3,7)./50
    dense=vcat(core.dense,[H2Matrices._PlanDense(D,1:20,1:10),H2Matrices._PlanDense(D2,9:11,5:11)])
    overlap=H2CompactMatvecPlan(core.shape,core.rows,core.cols,core.couplings,dense,core.rowcoeff,core.colcoeff,core.rowbuffer,core.colbuffer,core.rowperm,core.colperm)
    M=Matrix(C);M[core.rowperm[1:20],core.colperm[1:10]].+=D;M[core.rowperm[9:11],core.colperm[5:11]].+=D2
    for workers in (1,4)
        P=H2PacketMatvecPlan(overlap;workers)
        @test P*x ≈ M*x rtol=1e-12 atol=1e-12
        @test adjoint(P)*z ≈ M'*z rtol=1e-12 atol=1e-12
    end
    @test_throws ArgumentError H2PacketMatvecPlan(C;workers=0)
    C.global_index=true;P=H2PacketMatvecPlan(C);x=randn(rng,171);y=zeros(235)
    function allocations(A,x,y)
        mul!(y,A,x)
        @allocated mul!(y,A,x)
    end
    if VERSION>=v"1.10"
        @test allocations(P,x,y)==0
        @test allocations(adjoint(P),randn(rng,235),zeros(171))==0
        @test allocations(P,randn(rng,171,5),zeros(235,5))==0
        @test allocations(adjoint(P),randn(rng,235,3),zeros(171,3))==0
    end
end
@testset "Packet products with several right-hand sides" begin
    rng=MersenneTwister(921)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:203]
    Y=[Point2D(rand(rng),rand(rng)) for _ in 1:158]
    K=KernelMatrix(X,Y) do x,y
        exp(-sum(abs2,x-y))*(1+x[1]-0.3y[2])*cos(14*(x[1]*y[2]+x[2]*y[1]))
    end
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=8));ct=ClusterTree(copy(Y),GeometricSplitter(;nmax=8))
    H=assemble_hmatrix(K,rt,ct;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=203,strict=true,_print=false)
    for gi in (true,false),workers in (1,4)
        C.global_index=gi;P=H2PacketMatvecPlan(C;workers);M=Matrix(C;global_index=gi)
        for k in (0,1,2,3,5,9,17,33)
            Xr=randn(rng,158,k);Zr=randn(rng,203,k)
            for (A,input,expected) in ((P,Xr,M*Xr),(transpose(P),Zr,M'*Zr),(adjoint(P),Zr,M'*Zr))
                out=A*input
                @test size(out)==size(expected)
                @test out ≈ expected rtol=1e-10 atol=1e-11
                # Agrees with column-wise vector products up to rounding.
                for v in 1:k
                    @test out[:,v] ≈ A*input[:,v] rtol=1e-13 atol=1e-14
                end
                Yr=randn(rng,size(expected)...);old=copy(Yr)
                mul!(Yr,A,input,1.7,-0.3)
                @test Yr ≈ 1.7expected-0.3old rtol=1e-10 atol=1e-11
                fill!(Yr,NaN);mul!(Yr,A,input,0.,0.);@test all(iszero,Yr)
                k>0 && @test_throws DimensionMismatch mul!(zeros(size(expected,1),k+1),A,input)
                @test_throws DimensionMismatch mul!(zeros(1,k),A,input)
            end
            # Non-contiguous inputs and outputs.
            Xv=view(randn(rng,158,2k+1),:,1:2:2k);Yv=view(zeros(203,2k+1),:,2:2:2k+1)
            mul!(Yv,P,Xv);@test Yv ≈ M*Xv rtol=1e-10 atol=1e-11
        end
        Xr=randn(rng,158,4);Zr=randn(rng,203,4)
        @test dot(Zr,P*Xr) ≈ dot(adjoint(P)*Zr,Xr) rtol=1e-12 atol=1e-12
        # Workspace is private to each copy; numeric data is shared.
        Q=copy(P);Q*Xr
        @test Q.multi!==P.multi && Q.multi.rowcoeff!==P.multi.rowcoeff
        f=Threads.@spawn P*Xr
        g=Threads.@spawn adjoint(Q)*Zr
        @test fetch(f) ≈ M*Xr rtol=1e-10 atol=1e-11
        @test fetch(g) ≈ M'*Zr rtol=1e-10 atol=1e-11
        @test H2Matrices.multi_workspace_bytes(P,1)==0
        @test H2Matrices.multi_workspace_bytes(P,9)>0
        # Releasing the workspace frees it and shrinks it on the next use.
        R=copy(P);@test multi_workspace_bytes(R)==0
        X33=randn(rng,158,33);Y33=R*X33
        wide=multi_workspace_bytes(R)
        @test wide==multi_workspace_bytes(R,33)>multi_workspace_bytes(R,3)
        @test release_multi_workspace!(R)==wide && multi_workspace_bytes(R)==0
        @test R*Xr==P*Xr && adjoint(R)*Zr==adjoint(P)*Zr
        @test multi_workspace_bytes(R)==multi_workspace_bytes(R,4)
        @test R*X33==Y33
        @test release_multi_workspace!(R)>0 && release_multi_workspace!(R)==0
        x1=randn(rng,158);@test R*x1==P*x1
    end
    # Multi-vector products are also bitwise independent of the worker count.
    C.global_index=true;core=H2CompactMatvecPlan(C)
    P1=H2PacketMatvecPlan(core;workers=1);P4=H2PacketMatvecPlan(core;workers=4)
    Xr=randn(rng,158,7);Zr=randn(rng,203,7)
    @test P4*Xr==P1*Xr
    @test adjoint(P4)*Zr==adjoint(P1)*Zr
end
@testset "Packet plans without couplings or without near field" begin
    rng=MersenneTwister(922)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:40]
    K=KernelMatrix(X,X) do x,y
        exp(-sum(abs2,x-y))
    end
    # One leaf: the whole operator is a single dense block.
    T=ClusterTree(copy(X),GeometricSplitter(;nmax=64))
    H=assemble_hmatrix(K,T,T;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=40,strict=true,_print=false);M=Matrix(C)
    for workers in (1,3)
        P=H2PacketMatvecPlan(C;workers);x=randn(rng,40);Xr=randn(rng,40,3)
        @test isempty(P.packets) && !isempty(P.nearpackets)
        @test P*x ≈ M*x rtol=1e-12 atol=1e-13
        @test adjoint(P)*x ≈ M'*x rtol=1e-12 atol=1e-13
        @test P*Xr ≈ M*Xr rtol=1e-12 atol=1e-13
        @test adjoint(P)*Xr ≈ M'*Xr rtol=1e-12 atol=1e-13
    end
    # Well-separated point sets: an admissible root block, no near field.
    Yp=[Point2D(5+rand(rng),rand(rng)) for _ in 1:30]
    K2=KernelMatrix((x,y)->inv(norm(x-y)),X,Yp)
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=8));ct=ClusterTree(copy(Yp),GeometricSplitter(;nmax=8))
    H2=assemble_hmatrix(K2,rt,ct;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C2=compress_hmatrix_to_h2(H2;rtol=1e-11,maxrank=40,strict=true,_print=false);M2=Matrix(C2)
    for workers in (1,3)
        P=H2PacketMatvecPlan(C2;workers);x=randn(rng,30);z=randn(rng,40)
        @test isempty(P.nearpackets)
        @test P*x ≈ M2*x rtol=1e-11 atol=1e-13
        @test adjoint(P)*z ≈ M2'*z rtol=1e-11 atol=1e-13
        @test P*[x x] ≈ M2*[x x] rtol=1e-11 atol=1e-13
    end
end
