"""
    H2SolveResult

Result returned by `solve_cg` and `solve_gmres`.
"""
struct H2SolveResult{T}
    x::Vector{T}
    converged::Bool
    iterations::Int
    residuals::Vector{Float64}
end

function _apply_preconditioner(M, r)
    M === nothing && return copy(r)
    M isa AbstractMatrix && return M \ r
    return M(r)
end

"""
    solve_cg(A, b; tol=1e-6, maxiter=size(A,2), M=nothing, x0=zeros)

Solve `A*x = b` with preconditioned conjugate gradients. `A` may be an
`H2Matrix` or any matrix-like object supporting `mul!`/`*`. `M` is an optional
left preconditioner supplied either as a matrix or as a function `M(r)`.
"""
function solve_cg(A, b::AbstractVector;
                  tol::Float64=1e-6,
                  maxiter::Int=size(A, 2),
                  M=nothing,
                  x0=zeros(eltype(b), length(b)))
    size(A, 1) == length(b) || throw(DimensionMismatch("A and b dimensions differ"))
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("CG requires a square operator"))

    x = Vector{eltype(b)}(copy(x0))
    r = Vector{eltype(b)}(b - A * x)
    z = _apply_preconditioner(M, r)
    p = copy(z)
    rz_old = real(dot(r, z))
    bnorm = max(norm(b), eps(Float64))
    residuals = Float64[norm(r) / bnorm]
    residuals[end] <= tol && return H2SolveResult(x, true, 0, residuals)

    for iter in 1:maxiter
        Ap = A * p
        denom = real(dot(p, Ap))
        if abs(denom) <= eps(Float64)
            return H2SolveResult(x, false, iter - 1, residuals)
        end
        α = rz_old / denom
        x .+= α .* p
        r .-= α .* Ap
        push!(residuals, norm(r) / bnorm)
        residuals[end] <= tol && return H2SolveResult(x, true, iter, residuals)

        z = _apply_preconditioner(M, r)
        rz_new = real(dot(r, z))
        if abs(rz_old) <= eps(Float64)
            return H2SolveResult(x, false, iter, residuals)
        end
        β = rz_new / rz_old
        p .= z .+ β .* p
        rz_old = rz_new
    end

    return H2SolveResult(x, false, maxiter, residuals)
end

"""
    solve_gmres(A, b; tol=1e-6, restart=30, maxiter=size(A,2), M=nothing, x0=zeros)

Solve `A*x = b` with restarted left-preconditioned GMRES. The preconditioner
`M`, when supplied, is applied to residuals and operator products.
"""
function solve_gmres(A, b::AbstractVector;
                     tol::Float64=1e-6,
                     restart::Int=min(30, length(b)),
                     maxiter::Int=size(A, 2),
                     M=nothing,
                     x0=zeros(eltype(b), length(b)))
    size(A, 1) == length(b) || throw(DimensionMismatch("A and b dimensions differ"))
    restart > 0 || throw(ArgumentError("restart must be positive"))

    x = Vector{eltype(b)}(copy(x0))
    bnorm = max(norm(_apply_preconditioner(M, b)), eps(Float64))
    residuals = Float64[]
    total_iter = 0

    while total_iter < maxiter
        r = _apply_preconditioner(M, b - A * x)
        β = norm(r)
        push!(residuals, β / bnorm)
        residuals[end] <= tol && return H2SolveResult(x, true, total_iter, residuals)

        kmax = min(restart, maxiter - total_iter)
        V = zeros(eltype(b), length(b), kmax + 1)
        H = zeros(eltype(b), kmax + 1, kmax)
        V[:, 1] .= r ./ β

        used = 0
        for j in 1:kmax
            w = _apply_preconditioner(M, A * view(V, :, j))
            for i in 1:j
                H[i, j] = dot(view(V, :, i), w)
                w .-= H[i, j] .* view(V, :, i)
            end
            H[j + 1, j] = norm(w)
            if H[j + 1, j] > eps(Float64)
                V[:, j + 1] .= w ./ H[j + 1, j]
            end

            e1 = zeros(eltype(b), j + 1)
            e1[1] = β
            y = H[1:(j + 1), 1:j] \ e1
            relres = norm(e1 - H[1:(j + 1), 1:j] * y) / bnorm
            push!(residuals, relres)
            used = j
            total_iter += 1

            if relres <= tol || H[j + 1, j] <= eps(Float64)
                x .+= V[:, 1:j] * y
                return H2SolveResult(x, relres <= tol, total_iter, residuals)
            end
        end

        e1 = zeros(eltype(b), used + 1)
        e1[1] = β
        y = H[1:(used + 1), 1:used] \ e1
        x .+= V[:, 1:used] * y
    end

    r = _apply_preconditioner(M, b - A * x)
    push!(residuals, norm(r) / bnorm)
    return H2SolveResult(x, residuals[end] <= tol, maxiter, residuals)
end
