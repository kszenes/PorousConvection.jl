# PorousConvection.jl
- Shared memory (used CUDA primitives)
- Removed superfluous fields (gradT*) and dTdt
  - Fused kernels for since dTdt removed

## Theory

### Introduction
In this project, we implement a 3D multi-XPU solver for the convection of a fluid due to temperature through a porous media. This is a procees that is of particular interest when modelling geophysics.

### Equations
The porous convection process was modelled using the following system of equations:

$$
\boldsymbol{q_D} = -\frac{k}{\eta}(\nabla p - \rho_0\alpha\boldsymbol{g}T)
$$
$$
\nabla\cdot\boldsymbol{q_D} = 0
$$
$$
\boldsymbol{q_T} = -\frac{\lambda}{\rho_0 c_p}\nabla T
$$
$$
\frac{\partial T}{\partial t} + \frac{1}{\varphi}\boldsymbol{q_D}\cdot\nabla T + \nabla\cdot\boldsymbol{q_T} = 0
$$

The first equation represents the mass balance using two approximations:
- Darcy's law: assumes a linear dependence between the mass flux and the pressure gradient.
- Boussinesq approximation which models buoyancy by neglecting all contributions of density differences which are not weighted by the gravitational term $g$. This ends up being a good approximation as the terms containing $g$ are dominant.

?The second equation represents the incompressibility of the fluid
The third equation models the heat flux using Fourier's law.
The last equation represents the energy equation.

### Numerical Methods

To accelerate the solver, we use the pseudo-transient method which adds inertial terms to the Darcy and temperature diffusion fluxes:

$$
\beta\frac{\partial p}{\partial\tau} + \nabla\cdot\boldsymbol{q_D} = 0
$$
$$
\frac{\partial T}{\partial \tau} + \frac{T-T_\mathrm{old}}{\mathrm{d}t} + \frac{1}{\varphi}\boldsymbol{q_D}\cdot\nabla T + \nabla\cdot\boldsymbol{q_T} = 0
$$

These are solved using a conservative finite-differences scheme on a staggered grid. Indeed, the scalars (pressure and temperature) are defined inside the cells while the fluxes are defined on the cell boundaries.

### Implementation
The finte-difference is implemented using ParallelStencil Julia package and relies on ImplicitGlobalGrid for the distributed version.

The method consists of essentially 5 kernels for computing the various fields (as well as 2 additional kernels for implementing the boundary conditions). The 5 kernels compute the following fiels:
- Pressure kernels:
  - `compute_flux_p_3D!`: Darcy Flux
  - `update_Pf_3D!`: Updates pressure using the Darcy flux
- Temperature kernels:
  - `compute_flux_T_3D!`: Computes the Temperature flux
  - `computedTdt_3D!`: Computes the dTdt term
  - `update_T_3D!`: Updates the temperature using the results of the two previous kernels

In order to improve the performance of these kernels, we add shared memory for the 3 `compute_*` kernels. This reduces the number of global memory accesses. Since stencil computations are inherently memory bound, this optimization can greatly increase the throughput of our implementation.

In each kernel, only a single field is moved into shared memory. As we will be choosing the maximum block size (32, 4, 4) for our implementation, shared memory becomes a scarce ressource and storing additional fields into shared memory would decrease significantly the occupancy of the GPU. Thus the `update_*` kernels were not implemented using shared memoru.

In addition, auxiliary fields that appear in multiple kernels (such as `gradT*`) are recomputed in each respective kernel in order to also decrease the number of global memory accesses. In the context of stencil computation, we can afford making redundant calculation due to the arithmetic intensity being low.

### Code Organisation

```
PorousConvection.jl
├── PorousConvection        
    ├── img                  <-- Benchmark plots
    ├── scripts              <-- Execution scripts
        ├── benchmarks       <-- Scripts to reproduce benchmarks

    ├── src                  <-- Implementation source
    ├── test                 <-- Unit tests

└── matlab                 <-- Reference Matlab implementation
```

## Results

**OUTPUT PLOTS**

### Performance

The benchmarks were performed on Piz Daint supercomputer which is a single NVIDIA® Tesla® P100 with 16GB of memory per node.

#### Block Size
The performance characteristic of GPU kernels are highly dependent on correct selection of the block size. In the following benchmark, we evaluate the total runtime of the 5 stencil kernels for varying block sizes. This benchmark can be reproduced using the `PorousConvection/scripts/benchmarks/benchmark_block_size.jl` script.

Thus we select the (32, 4, 4) block size for all subsequent numerical experiments  as this exhibited the best performance.

![block_size](img/block_size_benchmark.png)

#### Speedup
In this section, we evaluate the speedup achieved in our improved implmentation using shared memory. The following benchmark compares the runtime for each kernel of the 5 kernels between the original and shared memory implementation. The percentage in each bar plot signifies the percentage of speedup. This benchmark may be reproduced using the `PorousConvection/scripts/benchmarks/benchmark_shmem.jl` script.


![speedup](img/shared_vs_original_speedup.png)

As illustrated by the plot, we obtain good speedups for each kernel averaging a **14.2%** overall speedup. Surprisingly, we also observe speedups for the kernels updating the pressure and temperature, which do not use shared memory. However, these two stencils have been ported to the `@parallel_indices` implementation which enables us to manually select the block size. This is most likely the reason for the performance increas.


#### Throughput
In this section, we discuss the achieved effective memory throughput for the 5 stencils. The results are illustrated in the plot below.

![throughput](img/throughput.png)

This plots suggests that the two stencils which were not implemented using shared memory (`P` adn `T`) exhibit the best effective throughput. Thus, we would expect these kernels to be less efficient. An explanation for this could be that since these kernels are the simplest expression containing few repeated fields (each field is used only in one expression) which would make cause redundant global memory accesses. This makes them particularly performant even using a naive implementation.