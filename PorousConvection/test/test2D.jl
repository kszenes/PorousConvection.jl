using Test
using PorousConvection.stencil2D_Threads
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

include("../src/porous_convection_implicit_2D.jl")

@testset "2D Porous Convection solver" begin
    # unit test
    _dx = _dy = _β_dτ = 1
    n = 5
    Pf = @zeros(n, n)
    qDx = @zeros(n + 1, n)
    qDy = @zeros(n, n + 1)
    @parallel compute_Pf_2D!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
    @test all(zeros(n, n) .≈ Array(Pf))

    # reference test
    x_idx = [1, 8, 6]
    y_idx = [9, 27, 29]
    T_ref = [
        0.001051733244842692 -4.0908559113159555 -44.43870012898717
        0.0010369604726116536 -4.176762104913213 -44.589709628799184
        0.0010423632053353465 -4.140200930869315 -44.52630731581964
    ]
    T = porous_convection_implicit_2D(; nx=30, ny=30, nt=10)
    @test all(T_ref .≈ T[x_idx, y_idx])
end

ParallelStencil.ParallelStencil.@reset_parallel_stencil()
