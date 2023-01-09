# PorousConvection.jl
- Shared memory (used CUDA primitives)
- Removed superfluous fields (gradT*) and dTdt
  - Fused kernels for since dTdt removed
