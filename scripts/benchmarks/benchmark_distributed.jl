using ImplicitGlobalGrid

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    println("Running on GPU")
    @init_parallel_stencil(CUDA, Float64, 3)
    using PorousConvection.stencil3D_CUDA
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

T_eff, t_it = porous_convection_implicit_3D(;
    nz=255, nt=1, nvis=1000, debug=false, save=false
)
@show t_it

# T_eff, t_it = porous_convection_implicit_3D(nz=127, nt=100, nvis=1000, debug=false, save=false)
# @show T_eff

# function run_strong_scaling()
#     nzs = 16 * 2 .^ (1:5) .- 1
#     t_its = []
#     T_effs = []
#     for nz in nzs
#         T_eff, t_it = porous_convection_implicit_3D(nz=nz, nt=100, nvis=1000, debug=false, save=false)
#         GC.gc(true); CUDA.reclaim() # free up memory
#         push!(t_its, t_it)
#         push!(T_effs, T_eff)
#     end
#     p = plot(
#         nzs,
#         T_effs;
#         title="Single GPU Porous Convection Performance",
#         xlabel="nz = ny",
#         ylabel="T_eff [GB/s]",
#         lw=2,
#         legend=false,
#     )
#     savefig(p, "../docs/single_gpu_perf.png")
#     @show nzs
#     @show t_its
#     @show T_effs
#     return t_its
# end

# t_its = run_strong_scaling()

MPI.Finalize()
