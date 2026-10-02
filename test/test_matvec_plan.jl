@testset "Reusable flattened H2 matvec plan" begin
    rng=MersenneTwister(112)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:171]
    Y=[Point2D(rand(rng),rand(rng)) for _ in 1:123]
    K=KernelMatrix(X,Y) do x,y
        exp(-sum(abs2,x-y))*(1+x[1]-0.3y[2])
    end
    H=assemble_hmatrix(K,ClusterTree(copy(X),GeometricSplitter(;nmax=8)),
        ClusterTree(copy(Y),GeometricSplitter(;nmax=8));comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-10,maxrank=120,strict=true,_print=false)
    for global_index in (true,false)
        C.global_index=global_index
        P=H2MatvecPlan(C)
        D=Matrix(C;global_index)
        @test P.h2===C
        x=randn(rng,length(Y));z=randn(rng,length(X))
        LP=H2LowRankMatvecPlan(C;rtol=0.0)
        for A in (P,transpose(P),adjoint(P),LP,transpose(LP),adjoint(LP))
            input=(A===P || A===LP) ? x : z
            expected=(A===P || A===LP) ? D*input : D'*input
            @test A*input ≈ expected rtol=1e-12 atol=1e-12
            output=randn(rng,length(expected));old=copy(output)
            mul!(output,A,input,1.7,-0.3)
            @test output ≈ 1.7expected-0.3old rtol=1e-12 atol=1e-12
            fill!(output,NaN);mul!(output,A,input,0.0,0.0)
            @test all(iszero,output)
            @test_throws DimensionMismatch mul!(zeros(length(output)+1),A,input)
            # Repeated products must clear previous coefficients.
            @test A*input ≈ expected rtol=1e-12 atol=1e-12
        end
        @test dot(z,P*x) ≈ dot(adjoint(P)*z,x) rtol=1e-12 atol=1e-12
    end
end
@testset "Recompression visits descendants of rank-zero roots" begin
    rng=MersenneTwister(330)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:235]
    K=KernelMatrix(X,X) do x,y
        exp(-sum(abs2,x-y))
    end
    tree=ClusterTree(copy(X),GeometricSplitter(;nmax=8))
    H=assemble_hmatrix(K,tree,tree;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=120,strict=true,_print=false)
    D=Matrix(C);before=storage_bytes(C)
    @test C.row_basis.k==0
    @test_throws ArgumentError recompress!(C;rtol=1e-12,maxrank=1,strict=true)
    @test Matrix(C)==D
    recompress!(C;rtol=1e-5,maxrank=120,strict=true)
    @test storage_bytes(C)<before
    @test norm(Matrix(C)-D)/norm(D)<1e-4
    @test C.row_basis.k==0
    @test_throws ArgumentError recompress!(C;rtol=-1.0)
    @test_throws ArgumentError recompress!(C;maxrank=0)
end
@testset "Low-rank coupling plan" begin
    K,rt,ct=separated_laplace2d(128;seed=923)
    H=assemble_hmatrix(K,rt,ct;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=128,strict=true,_print=false)
    for gi in (true,false)
        C.global_index=gi
        P=H2LowRankMatvecPlan(C;rtol=1e-8)
        D=Matrix(C;global_index=gi)
        x=randn(128);z=randn(128)
        @test norm(P*x-D*x)/norm(D*x)<1e-6
        @test norm(adjoint(P)*z-D'*z)/norm(D'*z)<1e-6
        @test dot(z,P*x) ≈ dot(adjoint(P)*z,x) rtol=1e-12 atol=1e-12
        @test storage_bytes(P)<=storage_bytes(C)
        for A in (P,adjoint(P),transpose(P))
            expected=A*x;y=randn(128);old=copy(y)
            mul!(y,A,x,1.7,-0.3)
            @test y ≈ 1.7expected-0.3old rtol=1e-12 atol=1e-12
            fill!(y,NaN);mul!(y,A,x,0.0,0.0);@test all(iszero,y)
            @test_throws DimensionMismatch mul!(zeros(1),A,x)
        end
    end
    @test_throws ArgumentError H2LowRankMatvecPlan(C;rtol=NaN)
end
@testset "Factorized coupling matvec allocations" begin
    rng=MersenneTwister(1)
    X=[Point2D(rand(rng),rand(rng)) for _ in 1:400]
    K=KernelMatrix(X,X) do x,y
        exp(-sum(abs2,x-y))
    end
    rt=ClusterTree(copy(X),GeometricSplitter(;nmax=8))
    H=assemble_hmatrix(K,rt,rt;comp=PartialACA(;rtol=1e-12),threads=false,global_index=true)
    C=compress_hmatrix_to_h2(H;rtol=1e-11,maxrank=120,strict=true,_print=false)
    P=H2LowRankMatvecPlan(C;rtol=1e-8)
    @test any(b->b.R!==nothing,P.couplings)
    x=randn(rng,length(X));y=similar(x)
    @test norm(P*x-C*x)/norm(C*x)<1e-6
    function allocated_product(A,x,y)
        mul!(y,A,x)
        @allocated mul!(y,A,x)
    end
    if VERSION>=v"1.10" # Allocation elimination depends on compiler version.
        @test allocated_product(P,x,y)==0
        @test allocated_product(adjoint(P),x,y)==0
    end
end
