const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

function separated_laplace2d(n; seed=42, shift=2.0)
    Random.seed!(seed)
    src = [Point2D(rand(), rand()) for _ in 1:n]
    tgt = [Point2D(shift + rand(), rand()) for _ in 1:n]
    K = KernelMatrix(src, tgt) do x, y
        r = norm(x - y)
        r > 0 ? 1 / (4π * r) : 0.0
    end
    rowtree = ClusterTree(deepcopy(src), GeometricSplitter(; nmax=max(12, n ÷ 8)))
    coltree = ClusterTree(deepcopy(tgt), GeometricSplitter(; nmax=max(12, n ÷ 8)))
    return K, rowtree, coltree
end

function dense_kernel_matrix(K, n)
    A = Matrix{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        A[i, j] = K[i, j]
    end
    return A
end

function random_points2d(n; seed=123)
    Random.seed!(seed)
    [Point2D(2rand() - 1, 2rand() - 1) for _ in 1:n]
end

function kernel_matrix(points, kind::Symbol)
    KernelMatrix(points, points) do x, y
        r2 = sum(abs2, x - y)
        if kind === :newton
            r2 == 0 ? 0.0 : inv(sqrt(r2))
        elseif kind === :logarithmic
            r2 == 0 ? 0.0 : -0.5 * log(r2)
        elseif kind === :exponential
            exp(-r2)
        else
            error("unknown kernel kind $kind")
        end
    end
end

function dense_spd_problem(n; seed=7)
    Random.seed!(seed)
    pts = random_points2d(n; seed)
    A = Matrix{Float64}(I, n, n)
    for j in 1:n, i in 1:n
        r2 = sum(abs2, pts[i] - pts[j])
        A[i, j] += 0.05 * exp(-r2)
    end
    A = Symmetric(A) |> Matrix
    tree = ClusterTree(deepcopy(pts), GeometricSplitter(; nmax=max(8, n ÷ 8)))
    return A, tree
end
