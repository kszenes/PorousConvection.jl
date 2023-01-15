"""
Benchmark to compare performance of naive vs share memory implementation.
"""
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(CUDA, Float64, 3)
import PorousConvection.stencil3D_CUDA_original as OG
import PorousConvection.stencil3D_CUDA as SHMEM

using Printf, BenchmarkTools

@printf("Benchmarking Shared Memory Implementation\n")

nz = 255
nx, ny = 2 * (nz + 1) - 1, nz
@show nx, ny, nz

# physics
Ra = 1000
lx, ly, lz = 40.0, 20.0, 20.0
k_ηf = 1.0
αρg = 1.0
ΔT = 200.0
ϕ = 0.1
_ϕ = 1.0 / ϕ
λ_ρCp = 1 / Ra * (αρg * k_ηf * ΔT * lz / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
# numerics
ϵtol = 1e-6
# maxiter = 10
maxiter = 10max(nx, ny)
cfl = 1.0 / sqrt(3.1)
re_D = 4π
ncheck = ceil(2max(nx, ny, nz))
# derived numerics
dx = lx / nx
dy = ly / ny
dz = lz / nz
_dx, _dy, _dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
yc = LinRange(dy / 2, ly - dy / 2, ny)
zc = LinRange(-ly + dy / 2, -dy / 2, ny)
θ_dτ_D = max(lx, ly, lz) / re_D / cfl / min(dx, dy, dz)
β_dτ_D = (re_D * k_ηf) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
_β_dτ_D = 1.0 ./ β_dτ_D
_1_θ_dτ_D = 1.0 ./ (1.0 + θ_dτ_D)
# --- array initialisation ---
# Pressure
Pf = @zeros(nx, ny, nz)
qDx = @zeros(nx + 1, ny, nz)
qDy = @zeros(nx, ny + 1, nz)
qDz = @zeros(nx, ny, nz + 1)
# Temperature
T = Data.Array([
    ΔT * exp(-xc[ix]^2 - (yc[iy] - ly / 2)^2 - (zc[iz] + lz / 2)^2) for ix in 1:nx,
    iy in 1:ny, iz in 1:nz
])
T[:, :, 1] .= ΔT / 2
T[:, :, end] .= -ΔT / 2
@parallel (1:size(T, 2), 1:size(T, 3)) OG.bc_xz!(T)
@parallel (1:size(T, 1), 1:size(T, 3)) OG.bc_yz!(T)
T_old = @zeros(nx, ny, nz)
qTx = @zeros(nx - 1, ny - 2, nz - 2)
qTy = @zeros(nx - 2, ny - 1, nz - 2)
qTz = @zeros(nx - 2, ny - 2, nz - 1)
dTdt = @zeros(nx - 2, ny - 2, nz - 2)
r_Pf = zeros(nx, ny, nz)
r_T = zeros(nx - 2, ny - 2, nz - 2)
gradTx = @zeros(nx - 1, ny - 2, nz - 2)
gradTy = @zeros(nx - 2, ny - 1, nz - 2)
gradTz = @zeros(nx - 2, ny - 2, nz - 1)
dt = 0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
_dt = 1.0 / dt
# implicit temperature params
re_T = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
θ_dτ_T = max(lx, ly, lz) / re_T / cfl / min(dx, dy, dz)
_1_θ_dτ_T = 1.0 / (1.0 + θ_dτ_T)
β_dτ_T = (re_T * λ_ρCp) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
_1_dt_β_dτ_T = 1.0 / (1.0 / dt + β_dτ_T)

@printf("=== Shared Memory ===\n")
threads = (32, 4, 4)
blocks = (size(T) .+ threads .- 1) .÷ threads

base_size = prod(size(Pf)) * sizeof(Float64) * 1e-9
kernel_rw = [8, 5, 7, 6, 6] # Read/Writes
A_effs = kernel_rw * base_size
T_effs = []
timings = []
# --- Pressure ---
time = @belapsed begin
    @parallel $blocks $threads shmem = (prod($threads .+ 1)) * sizeof(eltype($Pf)) SHMEM.compute_flux_p_3D!(
        $qDx, $qDy, $qDz, $Pf, $T, $k_ηf, $_dx, $_dy, $_dz, $_1_θ_dτ_D, $αρg
    )
end
push!(timings, time)
push!(T_effs, A_effs[1] / time)
@printf("compute_flux_p_3D!: %f [s] (%f [GB/s])\n", time, T_effs[1])
time = @belapsed begin
    @parallel $blocks $threads SHMEM.update_Pf_3D!(
        $Pf, $qDx, $qDy, $qDz, $_dx, $_dy, $_dz, $_β_dτ_D
    )
end
push!(timings, time)
push!(T_effs, A_effs[2] / time)
@printf("update_Pf_3D!: %f [s] (%f [GB/s])\n", time, T_effs[2])

# --- Temperature ---
time = @belapsed begin
    @parallel $blocks $threads shmem = prod($threads .+ 1) * sizeof(eltype($T)) SHMEM.compute_flux_T_3D!(
        $T, $qTx, $qTy, $qTz, $λ_ρCp, $_dx, $_dy, $_dz, $_1_θ_dτ_T
    )
end
push!(timings, time)
push!(T_effs, A_effs[3] / time)
@printf("compute_flux_T_3D!: %f [s] (%f GB/s)\n", time, T_effs[3])

time = @belapsed begin
    @parallel $blocks $threads shmem = prod($threads .+ 2) * sizeof(eltype($T)) SHMEM.computedTdt_3D!(
        $dTdt, $T, $T_old, $qDx, $qDy, $qDz, $_dx, $_dy, $_dz, $_dt, $_ϕ
    )
end
push!(timings, time)
push!(T_effs, A_effs[4] / time)
@printf("computedTdt!: %f [s] (%f GB/s)\n", time, T_effs[4])

time = @belapsed begin
    @parallel $blocks $threads SHMEM.update_T_3D!(
        $T, $dTdt, $qTx, $qTy, $qTz, $_dx, $_dy, $_dz, $_1_dt_β_dτ_T
    )
end
push!(timings, time)
push!(T_effs, A_effs[5] / time)
@printf("update_T_3D!: %f [s] (%f GB/s)\n", time, T_effs[5])

@printf("\n=== Original ===\n")
# --- Pressure ---
time = @belapsed begin
    @parallel OG.compute_flux_p_3D!(
        $qDx, $qDy, $qDz, $Pf, $T, $k_ηf, $_dx, $_dy, $_dz, $_1_θ_dτ_D, $αρg
    )
end
@printf("compute_flux_p_3D!: %f [s]\n", time)
time = @belapsed begin
    @parallel OG.update_Pf_3D!($Pf, $qDx, $qDy, $qDz, $_dx, $_dy, $_dz, $_β_dτ_D)
end
@printf("update_Pf_3D!: %f [s]\n", time)

# --- Temperature ---
time = @belapsed begin
    @parallel OG.compute_flux_T_3D!(
        $T,
        $qTx,
        $qTy,
        $qTz,
        $gradTx,
        $gradTy,
        $gradTz,
        $λ_ρCp,
        $_dx,
        $_dy,
        $_dz,
        $_1_θ_dτ_T,
    )
end
@printf("compute_flux_T_3D!: %f [s]\n", time)
time = @belapsed begin
    @parallel OG.computedTdt_3D!(
        $dTdt, $T, $T_old, $gradTx, $gradTy, $gradTz, $qDx, $qDy, $qDz, $_dt, $_ϕ
    )
end
@printf("computedTdt!: %f [s]\n", time)
time = @belapsed begin
    @parallel OG.update_T_3D!($T, $dTdt, $qTx, $qTy, $qTz, $_dx, $_dy, $_dz, $_1_dt_β_dτ_T)
end
@printf("update_T_3D!: %f [s]\n", time)
time = @belapsed begin
    @parallel (1:size($T, 2), 1:size($T, 3)) OG.bc_xz!($T)
end
