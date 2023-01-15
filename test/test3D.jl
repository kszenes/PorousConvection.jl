const USE_GPU = false
using Test
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    using PorousConvection.stencil3D_CUDA
else
    @init_parallel_stencil(Threads, Float64, 3)
    using PorousConvection.stencil3D_Threads
end

include("../src/porous_convection_implicit_3D.jl")

@testset "3D Porous Convection solver" begin
    # unit test
    _dx = _dy = _dz = _β_dτ = 1
    n = 6
    Pf = @zeros(n, n, n)
    qDx = @zeros(n + 1, n, n)
    qDy = @zeros(n, n + 1, n)
    qDz = @zeros(n, n, n + 1)
    threads = (2, 2, 2)
    blocks = (n, n, n) .÷ threads
    # Implement sharedMem
    @parallel blocks threads shmem = prod(threads .+ 2) * sizeof(eltype(Pf)) update_Pf_3D!(
        Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ
    )
    @test all(zeros(n, n) .≈ Array(Pf))

    # reference test
    x_idx = [15, 22, 24]
    y_idx = [25, 26, 15]
    z_idx = [18, 12, 16]

    T_ref = [
        1.7773234704935585e-5 1.6839829534796347e-6 37.36470327354834; -1.0810191286682869e-9 -1.0813183231026276e-9 2.273070758462484e-6; -1.0486481942070045e-9 -1.0472106668622358e-9 5.704107970008829e-10;;;
        1.5258926378859392e-6 1.7037607314630542e-7 1.2662183888109204; 1.1966641837048328e-8 1.1966357746585099e-8 1.071871024080536e-7; 1.1983848558511444e-8 1.198428890207378e-8 1.2065825083834645e-8;;;
        2.3300842146198643e-5 2.2496863664998135e-6 30.44308764177092; -1.1219045751634406e-12 -5.5545717453605076e-12 1.988096961260367e-6; -5.908819137959722e-12 -5.902848908280009e-12 1.526519110337282e-9
    ]
    T = porous_convection_implicit_3D(; nx=30, ny=30, nz=30, nt=10, save=false)
    @show T[x_idx, y_idx, z_idx]
    @show T_ref .≈ T[x_idx, y_idx, z_idx]
    @test all(T_ref .≈ T[x_idx, y_idx, z_idx])
end

ParallelStencil.ParallelStencil.@reset_parallel_stencil()
