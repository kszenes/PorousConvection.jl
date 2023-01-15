module PorousConvection

using ParallelStencil

# === 3D ===
include("stencil3D_CUDA_shmem.jl")
ParallelStencil.ParallelStencil.@reset_parallel_stencil()

include("stencil3D_CUDA_hide_comm.jl")
ParallelStencil.ParallelStencil.@reset_parallel_stencil()

include("stencil3D_Threads.jl")
ParallelStencil.ParallelStencil.@reset_parallel_stencil()

end
