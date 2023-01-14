module PorousConvection

using ParallelStencil

# === 2D ===
include("stencil2D_CUDA.jl")
ParallelStencil.ParallelStencil.@reset_parallel_stencil()

include("stencil2D_Threads.jl")
ParallelStencil.ParallelStencil.@reset_parallel_stencil()

# === 3D ===
include("stencil3D_CUDA.jl")
ParallelStencil.ParallelStencil.@reset_parallel_stencil()

include("stencil3D_CUDA_backup.jl")
ParallelStencil.ParallelStencil.@reset_parallel_stencil()

include("stencil3D_Threads.jl")
ParallelStencil.ParallelStencil.@reset_parallel_stencil()

end
