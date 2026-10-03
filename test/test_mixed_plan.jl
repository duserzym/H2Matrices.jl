@testset "Interleaved row storage and kernels" begin
    rng=MersenneTwister(77)
    for rows in 0:9, T in (Float64,Float32)
        A=randn(rng,rows,13);S=H2Matrices._InterleavedRows{T}(A)
        @test Matrix(S)==T.(A)
        x=randn(rng,13);z=randn(rng,rows+2);z0=copy(z)
        H2Matrices._il_forward!(z,1,S,x)
        @test z ≈ z0+[0;Float64.(T.(A))*x;0] rtol=1e-14 atol=1e-14
        w=randn(rng,rows+3);y=randn(rng,13);y0=copy(y)
        H2Matrices._il_adjoint!(y,S,w,2)
        @test y ≈ y0+Float64.(T.(A))'*w[3:rows+2] rtol=1e-14 atol=1e-14
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
    base=H2PacketMatvecPlan(C)
    for gi in (true,false)
        C.global_index=gi;compact=H2CompactMatvecPlan(C);M=Matrix(C;global_index=gi)
        P0=H2PacketMatvecPlan(compact)
        for rtol in (0.,1e-13,1e-8,1e-4),workers in (1,4)
            P=H2MixedPacketMatvecPlan(compact;workers,precision_rtol=rtol);s=precision_summary(P)
            @test !(:h2 in fieldnames(typeof(P)))
            @test s.bound <= rtol*s.reference_norm*(1+1e-12)
            @test s.storage_perturbation <= s.bound*(1+1e-12)
            rtol==0 && @test s.float32_rows==0 && storage_bytes(P)==storage_bytes(P0)
            rtol>=1e-8 && @test s.float32_rows>0 && storage_bytes(P)<storage_bytes(P0)
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
            end
            # The adjoint is the exact transpose of the same stored mixed operator.
            @test abs(dot(z,P*x)-dot(adjoint(P)*z,x)) <= 1e-13*norm(z)*norm(P*x)
            Q=copy(P)
            @test Q.rowcoeff!==P.rowcoeff && Q.partials[1]!==P.partials[1]
            @test all(b.hi===c.hi && b.lo===c.lo && b.hv===c.hv && b.scratch!==c.scratch for (b,c) in zip(P.packets,Q.packets))
            @test all(b.hi===c.hi && b.scratch!==c.scratch for (b,c) in zip(P.nearpackets,Q.nearpackets))
            f=Threads.@spawn P*x
            g=Threads.@spawn adjoint(Q)*z
            @test fetch(f) ≈ P*x rtol=1e-14
            @test fetch(g) ≈ adjoint(P)*z rtol=1e-14
        end
    end
    C.global_index=true;compact=H2CompactMatvecPlan(C)
    P=H2MixedPacketMatvecPlan(compact;precision_rtol=1e-8)
    @test precision_summary(P).rotated_packets>0
    @test any(b->!isempty(b.tau),P.packets)
    # A large budget stores every packet in Float32 without rotations.
    Pall=H2MixedPacketMatvecPlan(compact;precision_rtol=1e-3);sall=precision_summary(Pall)
    @test sall.float32_rows==sall.rows && sall.rotated_packets==0 && sall.rotation_bytes==0
    @test sum(b->length(b.hi.data)+length(b.lo.data),P.nearpackets)==sum(b->length(b.D),compact.dense)
    @test_throws ArgumentError H2MixedPacketMatvecPlan(compact;workers=0)
    @test_throws ArgumentError H2MixedPacketMatvecPlan(compact;precision_rtol=-1.)
    @test_throws ArgumentError H2MixedPacketMatvecPlan(compact;precision_rtol=NaN)
    function allocations(A,x,y)
        mul!(y,A,x)
        @allocated mul!(y,A,x)
    end
    if VERSION>=v"1.10"
        @test allocations(P,randn(rng,257),zeros(331))==0
        @test allocations(adjoint(P),randn(rng,331),zeros(257))==0
    end
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
    for rtol in (1e-10,1e-8),workers in (1,3)
        P=H2MixedPacketMatvecPlan(compact;workers,precision_rtol=rtol);s=precision_summary(P)
        @test s.absorbed_rotations>0
        @test s.bound <= rtol*s.reference_norm*(1+1e-12)
        x=randn(rng,600);z=randn(rng,600)
        @test norm(P*x-M*x) <= s.bound*norm(x)+1e-13*opnorm(M)*norm(x)
        @test norm(adjoint(P)*z-M'*z) <= s.bound*norm(z)+1e-13*opnorm(M)*norm(z)
        @test abs(dot(z,P*x)-dot(adjoint(P)*z,x)) <= 1e-13*norm(z)*norm(P*x)
    end
    # Absorption builds new basis nodes; the source compact plan is unchanged.
    @test Matrix(C)==M
    @test H2CompactMatvecPlan(C)*ones(600) == compact*ones(600)
    P=H2MixedPacketMatvecPlan(C;precision_rtol=1e-6)
    @test P*ones(600) ≈ M*ones(600) rtol=1e-5
end
