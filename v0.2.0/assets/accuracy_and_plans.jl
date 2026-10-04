using H2Matrices, LinearAlgebra, Random
using HMatrices: KernelMatrix, ClusterTree, GeometricSplitter, PartialACA, assemble_hmatrix
using StaticArrays

function accuracy_and_plans()
    BLAS.set_num_threads(1)
    rng = MersenneTwister(301)
    X = [SVector{2,Float64}(rand(rng), rand(rng)) for _ in 1:235]
    Y = [SVector{2,Float64}(rand(rng), rand(rng)) for _ in 1:171]
    K = KernelMatrix(X, Y) do x, y
        exp(-sum(abs2, x-y)) * (1+x[1]-0.3y[2]) * cos(16*(x[1]*y[2]+x[2]*y[1]))
    end
    rt = ClusterTree(copy(X), GeometricSplitter(; nmax=8))
    ct = ClusterTree(copy(Y), GeometricSplitter(; nmax=8))
    H = assemble_hmatrix(K, rt, ct;
        comp=PartialACA(; rtol=1e-12), global_index=true, threads=false)
    h2 = compress_hmatrix_to_h2(H;
        rtol=1e-10, maxrank=256, strict=true, _print=false)
    reference = Matrix(H)
    stored = Matrix(h2)
    conversion_error = norm(stored-reference)/norm(reference)
    @assert conversion_error < 1e-8
    x = randn(rng, size(h2,2))
    z = randn(rng, size(h2,1))
    compact = H2CompactMatvecPlan(h2)
    packet = H2PacketMatvecPlan(compact; workers=2)
    for plan in (H2MatvecPlan(h2), compact, packet)
        forward_error = norm(plan*x-stored*x)/norm(stored*x)
        adjoint_error = norm(adjoint(plan)*z-stored'*z)/norm(stored'*z)
        @assert forward_error < 1e-10 && adjoint_error < 1e-10
        @assert isapprox(dot(z,plan*x), dot(adjoint(plan)*z,x); rtol=1e-12, atol=1e-12)
        println(typeof(plan), ": forward=", forward_error, ", adjoint=", adjoint_error)
    end
    callers = [copy(packet) for _ in 1:2]
    inputs = [randn(rng, size(packet,2)) for _ in callers]
    outputs = [zeros(size(packet,1)) for _ in callers]
    @sync for i in eachindex(callers)
        Threads.@spawn mul!(outputs[i], callers[i], inputs[i])
    end
    for i in eachindex(callers)
        @assert isapprox(outputs[i], stored*inputs[i]; rtol=1e-10, atol=1e-11)
    end
    println("H-to-H2 relative error: ",conversion_error)
    println("Numeric bytes: H2=",storage_bytes(h2),", compact=",storage_bytes(compact),", packet=",storage_bytes(packet))
    println("Independent concurrent callers passed on Julia ",VERSION)
end
accuracy_and_plans()
