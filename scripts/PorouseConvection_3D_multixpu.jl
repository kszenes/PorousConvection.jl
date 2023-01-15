using ImplicitGlobalGrid

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    println("Running on GPU")
    @init_parallel_stencil(CUDA, Float64, 3)
    import PorousConvection.stencil3D_CUDA_hide_comm as OG
    import PorousConvection.stencil3D_CUDA_shmem as SHMEM
else
    println("Running on CPU")
    @init_parallel_stencil(Threads, Float64, 3)
    using PorousConvection.stencil3D_Threads
end

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

include("../src/porous_convection3D_xpu.jl")

# Task 1
porous_convection_implicit_3D()
# Task 2
# porous_convection_implicit_3D(; nz=127, nvis=100, nt=2000, save=true)
