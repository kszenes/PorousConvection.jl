module stencil3D_CUDA

export compute_flux_p_3D!,
    compute_Pf_3D!,
    compute_pressure_3D!,
    compute_flux_T_3D!,
    computedTdt_3D!,
    update_T_3D!,
    bc_xz!,
    bc_yz!,
    compute_temp_3D!

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(CUDA, Float64, 3)

macro all_own(A)
    esc(:($A[ix,iy,iz]))
end
macro d_xi_own(A)
    esc(:($A[ix+1, iy+1, iz+1] - $A[ix, iy+1, iz+1]))
end
macro d_yi_own(A)
    esc(:($A[ix+1, iy+1, iz+1] - $A[ix+1, iy, iz+1]))
end
macro d_zi_own(A)
    esc(:($A[ix+1, iy+1, iz+1] - $A[ix+1, iy+1, iz]))
end
macro d_xa_own(A)
    esc(:($A[ix+1, iy, iz] - $A[ix, iy, iz]))
end
macro d_ya_own(A)
    esc(:($A[ix, iy+1, iz] - $A[ix, iy, iz]))
end
macro d_za_own(A)
    esc(:($A[ix, iy, iz+1] - $A[ix, iy, iz]))
end
macro av_za_own(A)
    esc(:($A[ix, iy, iz+1]/2 + $A[ix, iy, iz]/2))
end

"""
Computes Darcy flux.
"""
@parallel_indices (ix, iy, iz) function compute_flux_p_3D!(
    qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ, αρg
)
    nx, ny, nz = size(Pf)

    tx = @threadIdx().x
    ty = @threadIdx().y
    tz = @threadIdx().z

    P_l = @sharedMem(eltype(Pf), (@blockDim().x+1, @blockDim().y+1, @blockDim().z+1))
    # Load data into shared memory
    P_l[tx,ty,tz] = Pf[ix,iy,iz]
    if (ix < nx && iy <= ny && iz <= nz)
        if (tx == @blockDim().x) P_l[tx+1,ty,tz] = Pf[ix+1,iy,iz] end
    end
    if (ix <= nx && iy < ny && iz <= nz)
        if (ty == @blockDim().y) P_l[tx,ty+1,tz] = Pf[ix,iy+1,iz] end
    end
    if (ix <= nx && iy <= ny && iz < nz)
        if (tz == @blockDim().z) P_l[tx,ty,tz+1] = Pf[ix,iy,iz+1] end
    end
    @sync_threads()

    if (ix < nx && iy <= ny && iz <= nz)
        qDx[ix+1,iy,iz] = qDx[ix+1,iy,iz] - _1_θ_dτ *
                            (qDx[ix+1,iy,iz] + k_ηf * _dx *
                            (P_l[tx+1,ty,tz]-P_l[tx,ty,tz]))
    end
    if (ix <= nx && iy < ny && iz <= nz)
        qDy[ix,iy+1,iz] = qDy[ix,iy+1,iz] - _1_θ_dτ *
                         (qDy[ix,iy+1,iz] + k_ηf * _dy *
                         (P_l[tx,ty+1,tz]-P_l[tx,ty,tz]))
    end
    if (ix <= nx && iy <= ny && iz < nz)
        qDz[ix,iy,iz+1] = qDz[ix,iy,iz+1] - _1_θ_dτ * (
                          qDz[ix,iy,iz+1] + k_ηf * _dz *
                          (P_l[tx,ty,tz+1]-P_l[tx,ty,tz]) -
                          αρg * @av_za_own(T)
            )
    end
    return nothing
end
"""
Updates pressure using Darcy flux.
"""
@parallel_indices (ix, iy, iz) function compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    nx, ny, nz = size(Pf)
    tx = @threadIdx().x
    ty = @threadIdx().y
    tz = @threadIdx().z

    qDx_l = CUDA.@cuDynamicSharedMem(eltype(qDx), (@blockDim().x+1, @blockDim().y, @blockDim().z))
    qDy_l = CUDA.@cuDynamicSharedMem(eltype(qDy),
        (@blockDim().x, @blockDim().y+1, @blockDim().z),
        sizeof(qDx_l))
    offset = sizeof(qDx_l) + sizeof(qDy_l)
    qDz_l = CUDA.@cuDynamicSharedMem(eltype(qDy),
        (@blockDim().x, @blockDim().y, @blockDim().z+1),
        offset)

    if (ix <= nx && iy <= ny && iz <= nz)
        qDx_l[tx,ty,tz] = qDx[ix,iy,iz]
        qDy_l[tx,ty,tz] = qDy[ix,iy,iz]
        qDz_l[tx,ty,tz] = qDz[ix,iy,iz]
        if (@threadIdx().x == @blockDim().x) qDx_l[tx+1,ty,tz] = qDx[ix+1,iy,iz] end
        if (@threadIdx().y == @blockDim().y) qDy_l[tx,ty+1,tz] = qDy[ix,iy+1,iz] end
        if (@threadIdx().z == @blockDim().z) qDz_l[tx,ty,tz+1] = qDz[ix,iy,iz+1] end
        @sync_threads()
        Pf[ix,iy,iz] = Pf[ix,iy,iz] - _β_dτ * (
                (qDx_l[tx+1,ty,tz] - qDx_l[tx,ty,tz]) * _dx +
                (qDy_l[tx,ty+1,tz] - qDy_l[tx,ty,tz]) * _dy +
                (qDz_l[tx,ty,tz+1] - qDz_l[tx,ty,tz]) * _dz
            )
    end
    return nothing
end

"""
Helper function which:
    1) Computes Darcy flux using `compute_flux_p_3D!()`
    2) Updates pressure accordingly using `compute_Pf_3D!()`
"""
function compute_pressure_3D!(
    Pf, T, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ, k_ηf, _1_θ_dτ, αρg
)
    # TODO: change threads values
    threads = (5, 5, 5)
    blocks = size(Pf) .÷ threads
    # TODO: slightly too large
    shmem = 3*(prod(threads.+1))*sizeof(eltype(Pf))
    @show size(Pf) size(qDx) threads blocks shmem
    @parallel blocks threads shmem=(prod(threads.+1))*sizeof(eltype(Pf)) compute_flux_p_3D!(qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ, αρg)
    @parallel blocks threads shmem=shmem compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    return nothing
end

"""
Compute pressure fluxes and gradients.
"""
@parallel_indices (ix, iy, iz) function compute_flux_T_3D!(
    T, qTx, qTy, qTz, gradTx, gradTy, gradTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
)
    if (ix <= size(qTx, 1) && iy <= size(qTx, 2) && iz <= size(qTx, 3))
        @all_own(qTx) = @all_own(qTx) - (@all_own(qTx) + λ_ρCp * _dx * @d_xi_own(T)) * _1_θ_dτ_T
        @all_own(gradTx) = @d_xi_own(T) * _dx
    end
    if (ix <= size(qTy, 1) && iy <= size(qTy, 2) && iz <= size(qTy, 3))
        @all_own(qTy) = @all_own(qTy) - (@all_own(qTy) + λ_ρCp * _dy * @d_yi_own(T)) * _1_θ_dτ_T
        @all_own(gradTy) = @d_yi_own(T) * _dy
    end
    if (ix <= size(qTz, 1) && iy <= size(qTz, 2) && iz <= size(qTz, 3))
        @all_own(qTz) = @all_own(qTz) - (@all_own(qTz) + λ_ρCp * _dy * @d_zi_own(T)) * _1_θ_dτ_T
        @all_own(gradTz) = @d_zi_own(T) * _dz
    end

    return nothing
end

"""
Compute dTdt expression.
"""
@parallel_indices (ix, iy, iz) function computedTdt_3D!(
    dTdt, T, T_old, gradTx, gradTy, gradTz, qDx, qDy, qDz, _dt, _ϕ
)
    nx, ny, nz = size(dTdt)
    if (ix <= nx && iy <= ny && iz <= nz)
        dTdt[ix, iy, iz] =
            (T[ix + 1, iy + 1, iz + 1] - T_old[ix + 1, iy + 1, iz + 1]) * _dt +
            (
                max(0.0, qDx[ix + 1, iy + 1, iz + 1]) * gradTx[ix, iy, iz] +
                min(0.0, qDx[ix + 2, iy + 1, iz + 1]) * gradTx[ix + 1, iy, iz] +
                max(0.0, qDy[ix + 1, iy + 1, iz + 1]) * gradTy[ix, iy, iz] +
                min(0.0, qDy[ix + 1, iy + 2, iz + 1]) * gradTy[ix, iy + 1, iz] +
                max(0.0, qDz[ix + 1, iy + 1, iz + 1]) * gradTz[ix, iy, iz] +
                min(0.0, qDz[ix + 1, iy + 1, iz + 2]) * gradTz[ix, iy, iz + 1]
            ) * _ϕ
    end
    return nothing
end

"""
Update temperature
"""
@parallel_indices (ix, iy, iz) function update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _β_dt)
    nx, ny, nz = size(T)
    if (ix < nx-1 && iy < ny-1 && iz < nz-1)
        T[ix+1, iy+1, iz+1] = T[ix+1, iy+1, iz+1] - _β_dt * (
            dTdt[ix, iy, iz] +
            (qTx[ix+1,iy,iz] - qTx[ix,iy,iz]) * _dx +
            (qTy[ix,iy+1,iz] - qTy[ix,iy,iz]) * _dy +
            (qTz[ix,iy,iz+1] - qTz[ix,iy,iz]) * _dz)
    end
    return nothing

end

"""
Apply von Neumann boundary conditions in xz-plane
"""
@parallel_indices (iy, iz) function bc_xz!(A)
    A[1, iy, iz] = A[2, iy, iz]
    A[end, iy, iz] = A[end - 1, iy, iz]
    return nothing
end

"""
Apply von Neumann boundary conditions in yz-plane
"""
@parallel_indices (ix, iz) function bc_yz!(A)
    A[ix, 1, iz] = A[ix, 2, iz]
    A[ix, end, iz] = A[ix, end - 1, iz]
    return nothing
end

"""
Helper function which:
    1) Computes temperature fluxes and gradients using `compute_flux_T_3D!()`
    2) Computes dTdt expression using `computedTdt_3D()`
    3) Updates temperature accordingly using `update_T_3D!()`
    4) Applies von Neumann boundary conditions using `bc_xz!()` and `bc_yz!()`
"""
function compute_temp_3D!(
    T,
    T_old,
    dTdt,
    qTx,
    qTy,
    qTz,
    gradTx,
    gradTy,
    gradTz,
    qDx,
    qDy,
    qDz,
    _dx,
    _dy,
    _dz,
    _dt,
    _1_dt_β_dτ_T,
    λ_ρCp,
    _1_θ_dτ_T,
    _ϕ,
)
    threads = (5, 5, 5)
    blocks = size(T) .÷ threads
    shmem = 3*(prod(threads.+1))*sizeof(eltype(T))
    @show size(T)

    @parallel blocks threads compute_flux_T_3D!(
        T, qTx, qTy, qTz, gradTx, gradTy, gradTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
    )
    @parallel blocks threads computedTdt_3D!(
        dTdt, T, T_old, gradTx, gradTy, gradTz, qDx, qDy, qDz, _dt, _ϕ
    )
    @parallel blocks threads shmem=shmem update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _1_dt_β_dτ_T)
    @parallel (1:size(T, 2), 1:size(T, 3)) bc_xz!(T)
    @parallel (1:size(T, 1), 1:size(T, 3)) bc_yz!(T)
    return nothing
end

end
