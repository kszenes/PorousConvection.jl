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

macro d_xif_own(A)
    esc(:($A[ix+1, iy+1, iz+1] - $A[ix, iy+1, iz+1]))
end
macro d_xib_own(A)
    esc(:($A[ix, iy+1, iz+1] - $A[ix-1, iy+1, iz+1]))
end
macro d_yif_own(A)
    esc(:($A[ix+1, iy+1, iz+1] - $A[ix+1, iy, iz+1]))
end
macro d_yib_own(A)
    esc(:($A[ix+1, iy, iz+1] - $A[ix+1, iy-1, iz+1]))
end
macro d_zif_own(A)
    esc(:($A[ix+1, iy+1, iz+1] - $A[ix+1, iy+1, iz]))
end
macro d_zib_own(A)
    esc(:($A[ix+1, iy+1, iz] - $A[ix+1, iy+1, iz-1]))
end
macro d_xaf_own(A)
    esc(:($A[ix+1, iy, iz] - $A[ix, iy, iz]))
end
macro d_xab_own(A)
    esc(:($A[ix, iy, iz] - $A[ix-1, iy, iz]))
end
macro d_yaf_own(A)
    esc(:($A[ix, iy+1, iz] - $A[ix, iy, iz]))
end
macro d_yab_own(A)
    esc(:($A[ix, iy, iz] - $A[ix, iy-1, iz]))
end
macro d_zaf_own(A)
    esc(:($A[ix, iy, iz+1] - $A[ix, iy, iz]))
end
macro d_zab_own(A)
    esc(:($A[ix, iy, iz] - $A[ix, iy, iz-1]))
end
macro av_zab_own(A)
    esc(:($A[ix, iy, iz]/2 + $A[ix, iy, iz-1]/2))
end

"""
Computes Darcy flux.
"""
@parallel_indices (ix, iy, iz) function compute_flux_p_3D!(
    qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ, αρg
)
    if (1 < ix < size(qDx, 1) && 1 <= iy <= size(qDx, 2) && 1 <= iz <= size(qDx, 3))
        qDx[ix,iy,iz] = qDx[ix,iy,iz] - _1_θ_dτ * (qDx[ix,iy,iz] + k_ηf * _dx * @d_xab_own(Pf))
    end
    if (1 <= ix <= size(qDy, 1) && 1 < iy < size(qDy, 2) && 1 <= iz <= size(qDy, 3))
        qDy[ix,iy,iz] = qDy[ix,iy,iz] - _1_θ_dτ * (qDy[ix,iy,iz] + k_ηf * _dy * @d_yab_own(Pf))
    end
    if (1 <= ix <= size(qDz, 1) && 1 <= iy <= size(qDz, 2) && 1 < iz < size(qDz, 3))
        qDz[ix,iy,iz] = qDz[ix,iy,iz] - _1_θ_dτ * (
                qDz[ix,iy,iz] + k_ηf * _dz * @d_zab_own(Pf) - αρg * @av_zab_own(T)
            )
    end
    return nothing
end
"""
Updates pressure using Darcy flux.
"""
@parallel_indices (ix, iy, iz) function compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    if (1 <= ix <= size(Pf, 1) && 1 <= iy <= size(Pf, 2) && 1 <= iz <= size(Pf, 3))
        Pf[ix,iy,iz] = Pf[ix,iy,iz] - _β_dτ * (
                @d_xaf_own(qDx) * _dx + @d_yaf_own(qDy) * _dy + @d_zaf_own(qDz) * _dz
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
    threads = (5, 5, 5)
    blocks = size(Pf) .÷ threads
    @show size(Pf) size(qDx) threads blocks
    @parallel blocks threads compute_flux_p_3D!(qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ, αρg)
    @parallel blocks threads compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    return nothing
end

"""
Compute pressure fluxes and gradients.
"""
@parallel_indices (ix, iy, iz) function compute_flux_T_3D!(
    T, qTx, qTy, qTz, gradTx, gradTy, gradTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
)
    if (1 <= ix <= size(qTx, 1) && 1 <= iy <= size(qTx, 2) && 1 <= iz <= size(qTx, 3))
        qTx[ix,iy,iz] = qTx[ix,iy,iz] - _1_θ_dτ_T * (qTx[ix,iy,iz] - λ_ρCp * _dx * @d_xif_own(T))
        gradTx[ix,iy,iz] = @d_xif_own(T) * _dx
    end
    if (1 <= ix <= size(qTy, 1) && 1 <= iy <= size(qTy, 2) && 1 <= iz <= size(qTy, 3))
        qTy[ix,iy,iz] = qTy[ix,iy,iz] - _1_θ_dτ_T * (qTy[ix,iy,iz] - λ_ρCp * _dy * @d_yif_own(T))
        gradTy[ix,iy,iz] = @d_yif_own(T) * _dy
    end
    if (1 <= ix <= size(qTz, 1) && 1 <= iy <= size(qTz, 2) && 1 <= iz <= size(qTz, 3))
        qTz[ix,iy,iz] = qTz[ix,iy,iz] - _1_θ_dτ_T * (qTz[ix,iy,iz] - λ_ρCp * _dz * @d_zif_own(T))
        gradTz[ix,iy,iz] = @d_zif_own(T) * _dz
    end

    # @all(qTx) = @all(qTx) - (@all(qTx) + λ_ρCp * _dx * @d_xi(T)) * _1_θ_dτ_T
    # @all(qTy) = @all(qTy) - (@all(qTy) + λ_ρCp * _dy * @d_yi(T)) * _1_θ_dτ_T
    # @all(qTz) = @all(qTz) - (@all(qTz) + λ_ρCp * _dz * @d_zi(T)) * _1_θ_dτ_T
    # @all(gradTx) = @d_xi(T) * _dx
    # @all(gradTy) = @d_yi(T) * _dy
    # @all(gradTz) = @d_zi(T) * _dz
    return nothing
end
# @parallel function compute_flux_T_3D!(
#     T, qTx, qTy, qTz, gradTx, gradTy, gradTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
# )
#     @all(qTx) = @all(qTx) - (@all(qTx) + λ_ρCp * _dx * @d_xi(T)) * _1_θ_dτ_T
#     @all(qTy) = @all(qTy) - (@all(qTy) + λ_ρCp * _dy * @d_yi(T)) * _1_θ_dτ_T
#     @all(qTz) = @all(qTz) - (@all(qTz) + λ_ρCp * _dz * @d_zi(T)) * _1_θ_dτ_T
#     @all(gradTx) = @d_xi(T) * _dx
#     @all(gradTy) = @d_yi(T) * _dy
#     @all(gradTz) = @d_zi(T) * _dz
#     return nothing
# end

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
@parallel function update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _β_dt)
    @inn(T) =
        @inn(T) -
        (@all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz) * _β_dt
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
    @show size(T) size(qTz)

    @parallel blocks threads compute_flux_T_3D!(
        T, qTx, qTy, qTz, gradTx, gradTy, gradTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
    )
    @parallel computedTdt_3D!(
        dTdt, T, T_old, gradTx, gradTy, gradTz, qDx, qDy, qDz, _dt, _ϕ
    )
    @parallel update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _1_dt_β_dτ_T)
    @parallel (1:size(T, 2), 1:size(T, 3)) bc_xz!(T)
    @parallel (1:size(T, 1), 1:size(T, 3)) bc_yz!(T)
    return nothing
end

end
