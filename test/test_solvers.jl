@testset "Solver wrappers" begin
    @testset "CG and GMRES on dense references" begin
        A, _ = dense_spd_problem(47; seed=70)
        b = randn(size(A, 1))

        cg = H2Matrices.solve_cg(A, b; tol=1e-8, maxiter=100)
        @test cg.converged
        @test norm(A * cg.x - b) / norm(b) < 1e-8
        @test cg.iterations <= 100

        gm = H2Matrices.solve_gmres(A, b; tol=1e-8, restart=12, maxiter=100)
        @test gm.converged
        @test norm(A * gm.x - b) / norm(b) < 1e-8
    end

    @testset "GMRES through H² matvec" begin
        A, tree = dense_spd_problem(60; seed=71)
        h2 = H2Matrices.compress_matrix_to_h2(A, tree, tree;
            rtol=1e-8,
            maxrank=30,
            global_index=false)
        b = randn(size(A, 1))

        gm = H2Matrices.solve_gmres(h2, b; tol=1e-6, restart=15, maxiter=120)
        @test gm.converged
        @test norm(Matrix(h2; global_index=false) * gm.x - b) / norm(b) < 1e-6
    end
end
