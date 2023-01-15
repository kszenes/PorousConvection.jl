module stencil3D_Threads

export compute_flux_p_3D!,
    update_Pf_3D!,
    compute_pressure_3D!,
    compute_flux_T_3D!,
    computedTdt_3D!,
    update_T_3D!,
    bc_xz!,
    bc_yz!,
    compute_temp_3D!

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3)

"""
Computes Darcy flux.

Memory transfers: 5 reads + 3 writes = 8 
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
    if (ix <= nx && iy <= ny && iz <= nz)
        P_l[tx,ty,tz] = Pf[ix,iy,iz]
        if (ix < nx && tx == @blockDim().x) P_l[tx+1,ty,tz] = Pf[ix+1,iy,iz] end
        if (iy < ny && ty == @blockDim().y) P_l[tx,ty+1,tz] = Pf[ix,iy+1,iz] end
        if (iz < nz && tz == @blockDim().z) P_l[tx,ty,tz+1] = Pf[ix,iy,iz+1] end
        @sync_threads()

        if (ix < nx)
            qDx[ix+1,iy,iz] = qDx[ix+1,iy,iz] - _1_θ_dτ *
                                (qDx[ix+1,iy,iz] + k_ηf * _dx *
                                (P_l[tx+1,ty,tz]-P_l[tx,ty,tz])
                            )
        end
        if (iy < ny)
            qDy[ix,iy+1,iz] = qDy[ix,iy+1,iz] - _1_θ_dτ * (
                                qDy[ix,iy+1,iz] + k_ηf * _dy *
                                (P_l[tx,ty+1,tz]-P_l[tx,ty,tz])
                            )
        end
        if (iz < nz)
            qDz[ix,iy,iz+1] = qDz[ix,iy,iz+1] - _1_θ_dτ * (
                                qDz[ix,iy,iz+1] + k_ηf * _dz *
                                (P_l[tx,ty,tz+1]-P_l[tx,ty,tz]) -
                                αρg * 0.5 * (T[ix,iy,iz+1] + T[ix,iy,iz])
                            )
        end
    end
    return nothing
end

"""
Updates pressure using Darcy flux.

Memory transfers: 4 reads + 1 writes = 5 
"""
@parallel_indices (ix, iy, iz) function update_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    nx, ny, nz = size(Pf)
    if (ix <= nx && iy <= ny && iz <= nz)
        Pf[ix,iy,iz] = Pf[ix,iy,iz] -   _β_dτ * (
            (qDx[ix+1,iy,iz] - qDx[ix,iy,iz]) * _dx +
            (qDy[ix,iy+1,iz] - qDy[ix,iy,iz]) * _dy +
            (qDz[ix,iy,iz+1] - qDz[ix,iy,iz]) * _dz)
    end
    return nothing
end

"""
Helper function which:
    1) Computes Darcy flux using `compute_flux_p_3D!()`
    2) Updates pressure accordingly using `update_Pf_3D!()`

Memory transfers total: 9 reads + 4 writes = 13 
"""
function compute_pressure_3D!(
    Pf, T, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ, k_ηf, _1_θ_dτ, αρg
)
    threads = (32, 4, 4)
    blocks  = (size(Pf) .+ threads .- 1) .÷ threads
    @parallel blocks threads shmem=(prod(threads.+1))*sizeof(eltype(Pf)) compute_flux_p_3D!(qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ, αρg)
    @parallel blocks threads update_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    return nothing
end

"""
Compute pressure fluxes and gradients.

Memory transfers: 4 reads + 3 writes = 7
"""
@parallel_indices (ix, iy, iz) function compute_flux_T_3D!(
    T, qTx, qTy, qTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
)
    nx, ny, nz = size(T)

    tx = @threadIdx().x + 1
    ty = @threadIdx().y + 1
    tz = @threadIdx().z + 1

    T_l = @sharedMem(eltype(T), (@blockDim().x+1, @blockDim().y+1, @blockDim().z+1))
    # Load data into shared memory
    if (ix <= nx - 1 && iy <= ny - 1 && iz <= nz - 1)
        T_l[tx,ty,tz] = T[ix+1,iy+1,iz+1]
        if (@threadIdx().x == 1) T_l[tx-1,ty,tz] = T[ix,iy+1,iz+1] end
        if (@threadIdx().y == 1) T_l[tx,ty-1,tz] = T[ix+1,iy,iz+1] end
        if (@threadIdx().z == 1) T_l[tx,ty,tz-1] = T[ix+1,iy+1,iz] end
        @sync_threads()

        if (ix <= nx-1 && iy <= ny-2 && iz <= nz-2)
            qTx[ix,iy,iz] = qTx[ix,iy,iz] -  _1_θ_dτ_T * (
                                qTx[ix,iy,iz] + λ_ρCp * _dx *
                                (T_l[tx,ty,tz] - T_l[tx-1,ty,tz])
                            )
            
        end
        if (ix <= nx-2 && iy <= ny-1 && iz <= nz-2)
            qTy[ix,iy,iz] = qTy[ix,iy,iz] - _1_θ_dτ_T * (
                                qTy[ix,iy,iz] + λ_ρCp * _dy *
                                (T_l[tx,ty,tz] - T_l[tx,ty-1,tz])
                            ) 
        end
        if (ix <= nx-2 && iy <= ny-2 && iz <= nz-1)
            qTz[ix,iy,iz] = qTz[ix,iy,iz] - _1_θ_dτ_T * (
                                qTz[ix,iy,iz] + λ_ρCp * _dz *
                                (T_l[tx,ty,tz] - T_l[tx,ty,tz-1])
                            ) 
        end
    end

    return nothing
end


"""
Compute dTdt expression.

Memory transfers: 5 reads + 1 writes = 6
"""
@parallel_indices (ix, iy, iz) function computedTdt_3D!(
    dTdt, T, T_old, qDx, qDy, qDz, _dx, _dy, _dz, _dt, _ϕ
)
    nx, ny, nz = size(T)
    tx = @threadIdx().x+1
    ty = @threadIdx().y+1
    tz = @threadIdx().z+1

    T_l = @sharedMem(eltype(T), (@blockDim().x+2, @blockDim().y+2, @blockDim().z+2))

    if (ix <= nx && iy <= ny && iz <= nz)
        T_l[tx, ty, tz] = T[ix, iy, iz]
        if (1 < ix < nx && 1 < iy < ny && 1 < iz < nz)
            if (@threadIdx().x == 1) T_l[tx-1,ty,tz] = T[ix-1,iy,iz] end
            if (@threadIdx().y == 1) T_l[tx,ty-1,tz] = T[ix,iy-1,iz] end
            if (@threadIdx().z == 1) T_l[tx,ty,tz-1] = T[ix,iy,iz-1] end
            if (@threadIdx().x == @blockDim().x) T_l[tx+1,ty,tz] = T[ix+1,iy,iz] end
            if (@threadIdx().y == @blockDim().y) T_l[tx,ty+1,tz] = T[ix,iy+1,iz] end
            if (@threadIdx().z == @blockDim().z) T_l[tx,ty,tz+1] = T[ix,iy,iz+1] end
            @sync_threads()
            dTdt[ix-1, iy-1, iz-1] =
                _dt * (T_l[tx, ty, tz] - T_old[ix, iy, iz]) +
                _ϕ * (
                    max(0.0, qDx[ix, iy, iz]) * (T_l[tx,ty,tz] - T_l[tx-1,ty,tz]) * _dx +
                    min(0.0, qDx[ix+1, iy, iz]) * (T_l[tx+1,ty,tz] - T_l[tx,ty,tz]) * _dx +
                    max(0.0, qDy[ix, iy, iz]) * (T_l[tx,ty,tz] - T_l[tx,ty-1,tz]) * _dy +
                    min(0.0, qDy[ix, iy+1, iz]) * (T_l[tx,ty+1,tz] - T_l[tx,ty,tz]) * _dy +
                    max(0.0, qDz[ix, iy, iz]) * (T_l[tx,ty,tz] - T_l[tx,ty,tz-1]) * _dz +
                    min(0.0, qDz[ix, iy, iz+1]) * (T_l[tx,ty,tz+1] - T_l[tx,ty,tz]) * _dz
                )
                # +
                # (qTx[ix,iy-1,iz-1] - qTx[ix-1,iy-1,iz-1]) * _dx +
                # (qTy[ix-1,iy,iz-1] - qTy[ix-1,iy-1,iz-1]) * _dy +
                # (qTz[ix-1,iy-1,iz] - qTz[ix-1,iy-1,iz-1]) * _dz
        end
    end
    return nothing
end

"""
Compute temparature

Memory transfers: 5 reads + 1 writes = 6
"""
@parallel_indices (ix, iy, iz) function update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _β_dt)
    nx, ny, nz = size(T)
    if (1 < ix < nx && 1 < iy < ny && 1 < iz < nz)
        T[ix,iy,iz] = T[ix,iy,iz] - _β_dt * (
            dTdt[ix-1,iy-1,iz-1]
            +
            (qTx[ix,iy-1,iz-1] - qTx[ix-1,iy-1,iz-1]) * _dx +
            (qTy[ix-1,iy,iz-1] - qTy[ix-1,iy-1,iz-1]) * _dy +
            (qTz[ix-1,iy-1,iz] - qTz[ix-1,iy-1,iz-1]) * _dz
            )
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

Memory transfers total: 14 reads + 5 writes = 19
"""
function compute_temp_3D!(
    T,
    T_old,
    dTdt,
    qTx,
    qTy,
    qTz,
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
    threads = (32, 4, 4)
    blocks  = (size(T) .+ threads .- 1) .÷ threads

    @parallel blocks threads shmem=prod(threads.+1)*sizeof(eltype(T)) compute_flux_T_3D!(
        T, qTx, qTy, qTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
    )
    @parallel blocks threads shmem=prod(threads.+2)*sizeof(eltype(T)) computedTdt_3D!(
        dTdt, T, T_old, qDx, qDy, qDz, _dx, _dy, _dz, _dt, _ϕ
    )
    @parallel blocks threads update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _1_dt_β_dτ_T)
    @parallel (1:size(T, 2), 1:size(T, 3)) bc_xz!(T)
    @parallel (1:size(T, 1), 1:size(T, 3)) bc_yz!(T)
    return nothing
end

end
