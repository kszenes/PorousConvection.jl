"""
Weak scaling benchmark comparing shared memory and comm/comp overlap implementation
"""
using ImplicitGlobalGrid

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    println("Running on GPU")
    @init_parallel_stencil(CUDA, Float64, 3)
    import PorousConvection.stencil3D_CUDA_original as OG
    import PorousConvection.stencil3D_CUDA as SHMEM
else
    println("Running on CPU")
    @init_parallel_stencil(Threads, Float64, 3)
    using PorousConvection.stencil3D_Threads
end

ENV["GKSwstype"] = "nul"

using Plots, Plots.Measures, Printf, Test, BenchmarkTools
default(;
    size=(600, 500),
    framestyle=:box,
    label=false,
    grid=false,
    margin=10mm,
    lw=2,
    labelfontsize=11,
    tickfontsize=11,
    titlefontsize=11,
)

include("../../src/porous_convection3D_xpu.jl")

@printf("Using shared memory\n")
T_eff, t_it = porous_convection_implicit_3D(;
    nz=255, nt=1, nvis=1000, debug=false, save=false, hide_comm=false
)
@show T_eff, t_it

@printf("Using communication/computation overlap\n")
T_eff, t_it = porous_convection_implicit_3D(;
    nz=255, nt=1, nvis=1000, debug=false, save=false, hide_comm=true
)
@show T_eff, t_it

MPI.Finalize()
