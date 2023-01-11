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
macro inn_own(A)
    esc(:($A[ix+1,iy+1,iz+1]))
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
    if (ix <= nx && iy <= ny && iz <= nz)
        P_l[tx,ty,tz] = Pf[ix,iy,iz]
        if (ix < nx && tx == @blockDim().x) P_l[tx+1,ty,tz] = Pf[ix+1,iy,iz] end
        if (iy < ny && ty == @blockDim().y) P_l[tx,ty+1,tz] = Pf[ix,iy+1,iz] end
        if (iz < nz && tz == @blockDim().z) P_l[tx,ty,tz+1] = Pf[ix,iy,iz+1] end
    end
    @sync_threads()

    if (ix < nx && iy <= ny && iz <= nz)
        qDx[ix+1,iy,iz] = qDx[ix+1,iy,iz] - _1_θ_dτ *
                            (qDx[ix+1,iy,iz] + k_ηf * _dx *
                            (P_l[tx+1,ty,tz]-P_l[tx,ty,tz])
                          )
    end
    if (ix <= nx && iy < ny && iz <= nz)
        qDy[ix,iy+1,iz] = qDy[ix,iy+1,iz] - _1_θ_dτ * (
                            qDy[ix,iy+1,iz] + k_ηf * _dy *
                            (P_l[tx,ty+1,tz]-P_l[tx,ty,tz])
                          )
    end
    if (ix <= nx && iy <= ny && iz < nz)
        qDz[ix,iy,iz+1] = qDz[ix,iy,iz+1] - _1_θ_dτ * (
                            qDz[ix,iy,iz+1] + k_ηf * _dz *
                            (P_l[tx,ty,tz+1]-P_l[tx,ty,tz]) -
                            αρg * 0.5 * (T[ix,iy,iz+1] + T[ix,iy,iz])
                          )
    end
    return nothing
end

# """
# Updates pressure using Darcy flux.
# TODO: Still does not work (only with threads = (5, 5, 5))
# """
# @parallel_indices (ix, iy, iz) function compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
#     nx, ny, nz = size(Pf)
#     tx = @threadIdx().x
#     ty = @threadIdx().y
#     tz = @threadIdx().z

#     qDx_l = CUDA.@cuDynamicSharedMem(eltype(qDx), (@blockDim().x+1, @blockDim().y, @blockDim().z))
#     qDy_l = CUDA.@cuDynamicSharedMem(eltype(qDy),
#         (@blockDim().x, @blockDim().y+1, @blockDim().z),
#         sizeof(qDx_l))
#     offset = sizeof(qDx_l) + sizeof(qDy_l)
#     qDz_l = CUDA.@cuDynamicSharedMem(eltype(qDz),
#         (@blockDim().x, @blockDim().y, @blockDim().z+1),
#         offset)

#     if (ix <= nx && iy <= ny && iz <= nz)
#         qDx_l[tx,ty,tz] = qDx[ix,iy,iz]
#         qDy_l[tx,ty,tz] = qDy[ix,iy,iz]
#         qDz_l[tx,ty,tz] = qDz[ix,iy,iz]
#         if (@threadIdx().x == @blockDim().x) qDx_l[tx+1,ty,tz] = qDx[ix+1,iy,iz] end
#         if (@threadIdx().y == @blockDim().y) qDy_l[tx,ty+1,tz] = qDy[ix,iy+1,iz] end
#         if (@threadIdx().z == @blockDim().z) qDz_l[tx,ty,tz+1] = qDz[ix,iy,iz+1] end
#         @sync_threads()
#         Pf[ix,iy,iz] = Pf[ix,iy,iz] - _β_dτ * (
#                 (qDx_l[tx+1,ty,tz] - qDx_l[tx,ty,tz]) * _dx +
#                 (qDy_l[tx,ty+1,tz] - qDy_l[tx,ty,tz]) * _dy +
#                 (qDz_l[tx,ty,tz+1] - qDz_l[tx,ty,tz]) * _dz
#             )
#     end
#     return nothing
# end

"""
Updates pressure using Darcy flux.
"""
@parallel function compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz) * _β_dτ
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
    threads = (8, 4, 4)
    # blocks = size(Pf) .÷ threads
    blocks  = (size(Pf) .+ threads .- 1) .÷ threads
    @parallel blocks threads shmem=(prod(threads.+1))*sizeof(eltype(Pf)) compute_flux_p_3D!(qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ, αρg)
    # threads = (5, 5, 5) # BUG: Only works with same number of threads in each dim
    # blocks  = (size(Pf) .+ threads .- 1) .÷ threads
    # @parallel blocks threads shmem=3*(prod(threads.+1))*sizeof(eltype(Pf)) compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    @parallel compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    return nothing
end

"""
Compute pressure fluxes and gradients.
"""
@parallel_indices (ix, iy, iz) function compute_flux_T_3D!(
    T, qTx, qTy, qTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
)
    nx, ny, nz = size(T)

    tx = @threadIdx().x
    ty = @threadIdx().y
    tz = @threadIdx().z

    T_l = @sharedMem(eltype(T), (@blockDim().x+1, @blockDim().y+1, @blockDim().z+1))
    # Load data into shared memory
    if (ix <= nx - 1 && iy <= ny - 1 && iz <= nz - 1)
        T_l[tx+1,ty+1,tz+1] = T[ix+1,iy+1,iz+1]
        T_l[tx,ty+1,tz+1] = T[ix,iy+1,iz+1]
        T_l[tx+1,ty+1,tz] = T[ix+1,iy+1,iz]
        T_l[tx+1,ty,tz+1] = T[ix+1,iy,iz+1]
    end
    @sync_threads()

    if (ix <= nx-1 && iy <= ny-2 && iz <= nz-2)
        qTx[ix,iy,iz] = qTx[ix,iy,iz] -  _1_θ_dτ_T * (
                            qTx[ix,iy,iz] + λ_ρCp * _dx *
                            (T_l[tx+1,ty+1,tz+1] - T_l[tx,ty+1,tz+1])
                        )
        
    end
    if (ix <= nx-2 && iy <= ny-1 && iz <= nz-2)
        qTy[ix,iy,iz] = qTy[ix,iy,iz] - _1_θ_dτ_T * (
                            qTy[ix,iy,iz] + λ_ρCp * _dy *
                            (T_l[tx+1,ty+1,tz+1] - T_l[tx+1,ty,tz+1])
                        ) 
    end
    if (ix <= nx-2 && iy <= ny-2 && iz <= nz-1)
        qTz[ix,iy,iz] = qTz[ix,iy,iz] - _1_θ_dτ_T * (
                            qTz[ix,iy,iz] + λ_ρCp * _dz *
                            (T_l[tx+1,ty+1,tz+1] - T_l[tx+1,ty+1,tz])
                        ) 
    end

    return nothing
end


"""
Compute dTdt expression.
"""
@parallel_indices (ix, iy, iz) function computedTdt_3D!(
    dTdt, T, T_old, qDx, qDy, qDz, qTx, qTy, qTz, _dx, _dy, _dz, _dt, _ϕ
)
    nx, ny, nz = size(T)
    tx = @threadIdx().x
    ty = @threadIdx().y
    tz = @threadIdx().z

    T_l = @sharedMem(eltype(T), (@blockDim().x+2, @blockDim().y+2, @blockDim().z+2))

    if (1 < ix < nx && 1 < iy < ny && 1 < iz < nz)
        T_l[tx, ty, tz] = T[ix,iy,iz]
        dTdt[ix-1, iy-1, iz-1] =
             _dt * (T[ix, iy, iz] - T_old[ix, iy, iz]) +  _ϕ * (
                max(0.0, qDx[ix, iy, iz]) * (T[ix,iy,iz] - T[ix-1,iy,iz]) * _dx +
                min(0.0, qDx[ix+1, iy, iz]) * (T[ix+1,iy,iz] - T[ix,iy,iz]) * _dx +
                max(0.0, qDy[ix, iy, iz]) * (T[ix,iy,iz] - T[ix,iy-1,iz]) * _dy +
                min(0.0, qDy[ix, iy+1, iz]) * (T[ix,iy+1,iz] - T[ix,iy,iz]) * _dy +
                max(0.0, qDz[ix, iy, iz]) * (T[ix,iy,iz] - T[ix,iy,iz-1]) * _dz +
                min(0.0, qDz[ix, iy, iz+1]) * (T[ix,iy,iz+1] - T[ix,iy,iz]) * _dz
            )
    end
    return nothing
end

@parallel_indices (ix, iy, iz) function update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _β_dt)
    nx, ny, nz = size(T)
    if (ix <= nx-2 && iy <= ny-2 && iz <= nz-2)
        T[ix+1,iy+1,iz+1] = T[ix+1,iy+1,iz+1] - _β_dt * (
            dTdt[ix,iy,iz]  +
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
    threads = (8, 4, 4)
    # blocks = size(T) .÷ threads
    blocks  = (size(T) .+ threads .- 1) .÷ threads

    @parallel blocks threads shmem=prod(threads.+1)*sizeof(eltype(T)) compute_flux_T_3D!(
        T, qTx, qTy, qTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
    )
    @parallel blocks threads shmem=prod(threads.+2)*sizeof(eltype(T)) computedTdt_3D!(
        dTdt, T, T_old, qDx, qDy, qDz, qTx, qTy, qTz, _dx, _dy, _dz, _dt, _ϕ
    )
    @parallel blocks threads update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _1_dt_β_dτ_T)
    @parallel (1:size(T, 2), 1:size(T, 3)) bc_xz!(T)
    @parallel (1:size(T, 1), 1:size(T, 3)) bc_yz!(T)
    return nothing
end

end
