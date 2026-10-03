@testset "Interleaved row storage and kernels" begin
    rng=MersenneTwister(77)
    for rows in 0:9, T in (Float64,Float32,H2Matrices._T48)
        A=randn(rng,rows,13).*exp10.(rand(rng,-8:2,rows,13));S=H2Matrices._InterleavedRows{T}(A)
        D=Float64.(Matrix(S))
        if T===H2Matrices._T48
            @test all(abs.(D.-A) .<= H2Matrices._U48 .* abs.(A))
            @test D!=A || rows==0
        else
            @test Matrix(S)==T.(A)
        end
        x=randn(rng,15);z=randn(rng,rows+2);z0=copy(z)
        H2Matrices._il_forward!(z,1,S,x,2)
        @test z ≈ z0+[0;D*x[3:15];0] rtol=1e-14 atol=1e-14
        w=randn(rng,rows+3);y=randn(rng,14);y0=copy(y)
        H2Matrices._il_adjoint!(y,1,S,w,2)
        @test y ≈ y0+[0;D'*w[3:rows+2]] rtol=1e-14 atol=1e-14
        # Several right-hand sides (strided blocks, every register-block width).
        for K in (1,2,3,5,9)
            X=randn(rng,2+13K);Z=randn(rng,1+(rows+1)K);Z0=copy(Z)
            H2Matrices._il_forward_k!(Z,1,rows+1,S,X,2,13,K)
            for v in 1:K
                @test Z[1+(v-1)*(rows+1)+1:1+(v-1)*(rows+1)+rows] ≈ Z0[1+(v-1)*(rows+1)+1:1+(v-1)*(rows+1)+rows]+D*X[2+(v-1)*13+1:2+v*13] rtol=1e-14 atol=1e-14
            end
            W=randn(rng,3+(rows+2)K);Y=randn(rng,1+14K);Y0=copy(Y)
            H2Matrices._il_adjoint_k!(Y,1,14,S,W,3,rows+2,K)
            for v in 1:K
                @test Y[1+(v-1)*14+1:1+(v-1)*14+13] ≈ Y0[1+(v-1)*14+1:1+(v-1)*14+13]+D'*W[3+(v-1)*(rows+2)+1:3+(v-1)*(rows+2)+rows] rtol=1e-14 atol=1e-14
            end
        end
    end
    # Householder rotations: forward and transposed application are inverse transposes.
    k=11;r=6;U=Matrix(qr(randn(rng,k,k)).Q)
    A,tau=LAPACK.geqrf!(U[:,1:r]);Q=LAPACK.ormqr!('L','N',A,tau,Matrix{Float64}(I,k,k))
    hv=reduce(vcat,[A[i+1:k,i] for i in 1:r])
    @test length(hv)==H2Matrices._reflector_entries(k,r)
    z=randn(rng,k);z1=copy(z);H2Matrices._apply_reflectors!(z1,hv,tau,false)
    @test z1 ≈ Q*z rtol=1e-14
    z2=copy(z);H2Matrices._apply_reflectors!(z2,hv,tau,true)
    @test z2 ≈ Q'*z rtol=1e-14
    w=randn(rng,k);w2=copy(w);H2Matrices._apply_reflectors!(w2,hv,tau,true)
    @test abs(dot(w,z1)-dot(w2,z)) <= 1e-14*norm(w)*norm(z)
    zz=randn(rng,k+3);zz1=copy(zz);H2Matrices._apply_reflectors!(zz1,2,k,hv,tau,false)
    @test zz1[3:k+2] ≈ Q*zz[3:k+2] rtol=1e-14
    @test zz1[[1,2,k+3]]==zz[[1,2,k+3]]
end
@testset "Mixed-precision packets obey the stored-operator bound" begin
    rng=MersenneTwister(1203)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:331]
    Y=[Point2D(rand(rng),rand(rng)) for _ in 1:257]
    K=KernelMatrix(X,Y) do x,y
        exp(-sum(abs2,x-y))*(1+x[1]-0.3y[2])*cos(16*(x[1]*y[2]+x[2]*y[1]))
    end
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=8));ct=ClusterTree(copy(Y),GeometricSplitter(;nmax=8))
    H=assemble_hmatrix(K,rt,ct;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=331,strict=true,_print=false)
    for gi in (true,false)
        C.global_index=gi;compact=H2CompactMatvecPlan(C);M=Matrix(C;global_index=gi)
        P0=H2PacketMatvecPlan(compact)
        for rtol in (0.,1e-13,1e-8,1e-4),workers in (1,4),f48 in (false,true)
            P=H2MixedPacketMatvecPlan(compact;workers,precision_rtol=rtol,format48=f48);s=precision_summary(P)
            f48 || @test s.float48_rows==0
            f48 && rtol==1e-13 && @test s.float48_rows>0 && storage_bytes(P)<storage_bytes(H2MixedPacketMatvecPlan(compact;precision_rtol=rtol))
            @test !(:h2 in fieldnames(typeof(P)))
            @test s.bound <= rtol*s.reference_norm*(1+1e-12)
            @test s.storage_perturbation <= s.bound*(1+1e-12)
            @test s.reference_norm ≈ norm(M) rtol=1e-10
            rtol==0 && @test s.float32_rows==0 && storage_bytes(P)==storage_bytes(P0)
            rtol>=1e-8 && @test s.float32_rows>0 && storage_bytes(P)<storage_bytes(P0)
            # The near field is stored as in the packet plan, in Float64.
            @test all(b.matrix==c.matrix for (b,c) in zip(P.engine.nearpackets,P0.nearpackets))
            x=randn(rng,size(M,2));z=randn(rng,size(M,1))
            # Rigorous: ‖(Ã-A)x‖ ≤ bound‖x‖, plus Float64 roundoff of both products.
            @test norm(P*x-M*x) <= s.bound*norm(x)+1e-13*opnorm(M)*norm(x)
            @test norm(adjoint(P)*z-M'*z) <= s.bound*norm(z)+1e-13*opnorm(M)*norm(z)
            rtol==0 && @test P*x ≈ P0*x rtol=1e-13
            for (A,input) in ((P,x),(transpose(P),z),(adjoint(P),z))
                expected=A*input
                y=randn(rng,length(expected));old=copy(y)
                mul!(y,A,input,1.7,-0.3)
                @test y ≈ 1.7expected-0.3old rtol=1e-13 atol=1e-13
                fill!(y,NaN);mul!(y,A,input,0.,0.);@test all(iszero,y)
                @test_throws DimensionMismatch mul!(zeros(1),A,input)
                # Several right-hand sides apply the same stored operator.
                Xm=randn(rng,length(input),9);Ym=A*Xm
                @test Ym ≈ reduce(hcat,[A*Xm[:,j] for j in 1:9]) rtol=1e-14 atol=1e-14
                Yo=randn(rng,size(Ym)...);Yold=copy(Yo);mul!(Yo,A,Xm,0.5,2.0)
                @test Yo ≈ 0.5Ym+2Yold rtol=1e-13 atol=1e-13
                @test_throws DimensionMismatch mul!(zeros(1,9),A,Xm)
            end
            # The adjoint is the exact transpose of the same stored mixed operator.
            @test abs(dot(z,P*x)-dot(adjoint(P)*z,x)) <= 1e-13*norm(z)*norm(P*x)
            Q=copy(P)
            @test Q.engine.rowcoeff!==P.engine.rowcoeff && Q.engine.slots!==P.engine.slots
            @test all(a!==b for (a,b) in zip(Q.engine.scratch,P.engine.scratch))
            @test Q.engine.packets===P.engine.packets && Q.engine.rows===P.engine.rows && Q.precision===s
            @test all(b.matrix===c.matrix for (b,c) in zip(P.engine.nearpackets,Q.engine.nearpackets))
            f=Threads.@spawn P*x
            g=Threads.@spawn adjoint(Q)*z
            @test fetch(f)==P*x
            @test fetch(g)==adjoint(P)*z
        end
    end
    C.global_index=true;compact=H2CompactMatvecPlan(C);M=Matrix(C)
    P=H2MixedPacketMatvecPlan(compact;precision_rtol=1e-8)
    @test precision_summary(P).rotated_packets>0
    @test any(b->!isempty(b.tau),P.engine.packets)
    # Products are bitwise independent of the worker count (single and several vectors).
    x=randn(rng,257);z=randn(rng,331);Xm=randn(rng,257,7);Zm=randn(rng,331,7)
    for w in (2,3,7)
        Pw=H2MixedPacketMatvecPlan(compact;workers=w,precision_rtol=1e-8)
        @test Pw*x==P*x && adjoint(Pw)*z==adjoint(P)*z
        @test Pw*Xm==P*Xm && adjoint(Pw)*Zm==adjoint(P)*Zm
    end
    # A large budget stores every packet in Float32 without rotations.
    Pall=H2MixedPacketMatvecPlan(compact;precision_rtol=1e-3);sall=precision_summary(Pall)
    @test sall.float32_rows==sall.rows && sall.rotated_packets==0 && sall.rotation_bytes==0
    @test sum(b->length(b.lo.data),Pall.engine.packets)==sum(b->length(b.S),compact.couplings)
    # Unaligned, overlapping near-field rows.
    D=fill(0.03,20,10)
    dense=vcat(compact.dense,[H2Matrices._PlanDense(D,3:22,1:10)])
    overlap=H2CompactMatvecPlan(compact.shape,compact.rows,compact.cols,compact.couplings,dense,compact.rowcoeff,compact.colcoeff,compact.rowbuffer,compact.colbuffer,compact.rowperm,compact.colperm)
    Mo=Matrix(C);Mo[compact.rowperm[3:22],compact.colperm[1:10]].+=D
    for workers in (1,4),rtol in (0.,1e-8)
        Po=H2MixedPacketMatvecPlan(overlap;workers,precision_rtol=rtol);so=precision_summary(Po)
        x=randn(rng,257);z=randn(rng,331)
        @test norm(Po*x-Mo*x) <= so.bound*norm(x)+1e-13*opnorm(Mo)*norm(x)
        @test norm(adjoint(Po)*z-Mo'*z) <= so.bound*norm(z)+1e-13*opnorm(Mo)*norm(z)
    end
    @test_throws ArgumentError H2MixedPacketMatvecPlan(compact;workers=0)
    @test_throws ArgumentError H2MixedPacketMatvecPlan(compact;precision_rtol=-1.)
    @test_throws ArgumentError H2MixedPacketMatvecPlan(compact;precision_rtol=NaN)
    @test_throws ArgumentError H2MixedPacketMatvecPlan(compact;precision_rtol=Inf)
    function allocations(A,x,y)
        mul!(y,A,x)
        @allocated mul!(y,A,x)
    end
    if VERSION>=v"1.10"
        @test allocations(P,randn(rng,257),zeros(331))==0
        @test allocations(adjoint(P),randn(rng,331),zeros(257))==0
        @test allocations(P,randn(rng,257,5),zeros(331,5))==0
        @test allocations(adjoint(P),randn(rng,331,3),zeros(257,3))==0
        P48=H2MixedPacketMatvecPlan(compact;precision_rtol=1e-13,format48=true)
        @test precision_summary(P48).float48_rows>0
        @test allocations(P48,randn(rng,257),zeros(331))==0
        @test allocations(adjoint(P48),randn(rng,331),zeros(257))==0
    end
    # Multi-vector workspace accounting and release.
    R=H2MixedPacketMatvecPlan(compact;precision_rtol=1e-8,workers=2);Xr=randn(rng,257,4)
    @test multi_workspace_bytes(R)==0
    Yr=R*Xr
    @test multi_workspace_bytes(R)==multi_workspace_bytes(R,4)>0
    @test release_multi_workspace!(R)>0 && multi_workspace_bytes(R)==0
    @test R*Xr==Yr
end
@testset "Mixed-precision rotations absorbed into explicit bases" begin
    rng=MersenneTwister(1821)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:600]
    K=KernelMatrix(X,X) do x,y
        r=norm(x-y)
        r>0 ? 1/r : 0.0
    end
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=16))
    H=assemble_hmatrix(K,rt,rt;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-10,maxrank=600,strict=true,_print=false)
    compact=H2CompactMatvecPlan(C);M=Matrix(C)
    @test any(n->!n.identity && !isempty(n.coeff),compact.rows)
    for rtol in (1e-10,1e-8),workers in (1,3),f48 in (false,true)
        P=H2MixedPacketMatvecPlan(compact;workers,precision_rtol=rtol,format48=f48);s=precision_summary(P)
        @test s.absorbed_rotations>0
        @test s.bound <= rtol*s.reference_norm*(1+1e-12)
        x=randn(rng,600);z=randn(rng,600)
        @test norm(P*x-M*x) <= s.bound*norm(x)+1e-13*opnorm(M)*norm(x)
        @test norm(adjoint(P)*z-M'*z) <= s.bound*norm(z)+1e-13*opnorm(M)*norm(z)
        @test abs(dot(z,P*x)-dot(adjoint(P)*z,x)) <= 1e-13*norm(z)*norm(P*x)
        Xm=randn(rng,600,6)
        @test P*Xm ≈ reduce(hcat,[P*Xm[:,j] for j in 1:6]) rtol=1e-14
        @test adjoint(P)*Xm ≈ reduce(hcat,[adjoint(P)*Xm[:,j] for j in 1:6]) rtol=1e-14
    end
    # Absorption builds new basis nodes; the source compact plan is unchanged.
    @test Matrix(C)==M
    @test H2CompactMatvecPlan(C)*ones(600) == compact*ones(600)
    P=H2MixedPacketMatvecPlan(C;precision_rtol=1e-6)
    @test P*ones(600) ≈ M*ones(600) rtol=1e-5
    # Construction is bitwise deterministic under threads: the absorbed
    # left (own packet) and right (parent packet) factors are applied in a
    # fixed order whatever the schedule.
    stored(P)=(io=IOBuffer();foreach(b->write(io,b.hi.data,b.lo.data,b.mid.data.hi,b.hv),P.engine.packets);
        foreach(n->write(io,n.E,n.V),P.engine.rows);take!(io))
    ref=stored(H2MixedPacketMatvecPlan(compact;precision_rtol=1e-10,format48=true))
    @test all(stored(H2MixedPacketMatvecPlan(compact;precision_rtol=1e-10,format48=true,workers=w))==ref for w in (1,2,4,1,2,4))
end
@testset "Mixed-precision plans on condensed conversions and consuming builds" begin
    rng=MersenneTwister(933)
    X=[Point3D(normalize(randn(rng,3))...) for _ in 1:600]
    K=KernelMatrix(X,X) do x,y
        r=norm(x-y)
        r>0 ? (1+0.1x[1])/r : 0.0
    end
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=16));ct=ClusterTree(copy(X),GeometricSplitter(;nmax=16))
    H=assemble_hmatrix(K,rt,ct;comp=PartialACA(;rtol=1e-11),global_index=true,threads=false)
    C=compress_hmatrix_to_h2(H;rtol=1e-9,maxrank=300,strict=true,_print=false)
    # The condensed conversion stores exact identity leaves for saturated bases.
    @test any(cb->H2Matrices.isleaf(cb) && cb.k==length(cb) && cb.V==I,H2Matrices.nodes(C.row_basis))
    M=Matrix(C);x=randn(rng,600);z=randn(rng,600)
    for rtol in (0.,1e-12,1e-8),f48 in (false,true),workers in (1,4)
        P=H2MixedPacketMatvecPlan(C;workers,precision_rtol=rtol,format48=f48);s=precision_summary(P)
        source=copy(C)
        Q=H2MixedPacketMatvecPlan(source;workers,precision_rtol=rtol,format48=f48,consume=true)
        @test all(l->l.uniform===nothing && l.dense===nothing,H2Matrices.leaves(source))
        @test precision_summary(Q)==s && storage_bytes(Q)==storage_bytes(P)
        @test P*x==Q*x && adjoint(P)*z==adjoint(Q)*z
        @test norm(P*x-M*x) <= s.bound*norm(x)+1e-13*opnorm(M)*norm(x)
        @test norm(adjoint(P)*z-M'*z) <= s.bound*norm(z)+1e-13*opnorm(M)*norm(z)
        rtol>=1e-8 && @test s.float32_rows>0
    end
end
@testset "Mixed-precision overflow guard" begin
    rng=MersenneTwister(1)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:600]
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=16))
    for scale in (1e38,1e45,1e300)
        K=KernelMatrix(X,X) do x,y
            r=norm(x-y)
            r>0 ? scale/r : 0.0
        end
        C=compress_hmatrix_to_h2(assemble_hmatrix(K,rt,rt;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true);
            rtol=1e-10,maxrank=600,strict=true,_print=false)
        M=Matrix(C);x=randn(rng,600)
        for rtol in (1e-13,1e-3),f48 in (false,true)
            P=H2MixedPacketMatvecPlan(C;precision_rtol=rtol,format48=f48);s=precision_summary(P)
            # Packets too large for Float32 (and, at 1e300, the overflowing η) stay out of the reduced formats.
            @test s.float32_rows==0 && s.float32_ineligible_packets==s.packets
            scale==1e300 && @test s.float48_rows==0
            y=P*x;yr=M*x
            @test all(isfinite,y) && all(isfinite,adjoint(P)*x)
            @test norm(y-yr) <= s.bound*norm(x)+1e-13*opnorm(M)*norm(x)
        end
    end
end
