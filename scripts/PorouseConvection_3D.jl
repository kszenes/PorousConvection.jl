const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    using PorousConvection.stencil3D_CUDA
else
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

include("../src/porous_convection_implicit_3D.jl")

nz = 63
porous_convection_implicit_3D(;
    nx=2 * (nz + 1) - 1, ny=nz, nz=nz, nvis=50, nt=500, save=true
)
# porous_convection_implicit_3D(; nx=255, ny=127, nz=127, nvis=50, nt=2000, save=true)
