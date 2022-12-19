const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    using PorousConvection.stencil2D_CUDA
else
    @init_parallel_stencil(Threads, Float64, 2)
    using PorousConvection.stencil2D_Threads
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

ENV["GKSwstype"] = "nul"
if isdir("viz_out") == false
    mkdir("viz_out")
end
loadpath = "./viz_out/";
anim = Animation(loadpath, String[]);
println("Animation directory: $(anim.dir)")

include("../src/porous_convection_implicit_2D.jl")

porous_convection_implicit_2D(; nx=511, ny=1023, nvis=50, nt=4000)
