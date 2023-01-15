module stencil3D_Threads_original

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
"""
@parallel function compute_flux_p_3D!(
    qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ, αρg
)
    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf * _dx * @d_xa(Pf)) * _1_θ_dτ
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf * _dy * @d_ya(Pf)) * _1_θ_dτ
    @inn_z(qDz) =
        @inn_z(qDz) - (@inn_z(qDz) + k_ηf * @d_za(Pf) * _dz - αρg * @av_za(T)) * _1_θ_dτ
    return nothing
end

"""
Updates pressure using Darcy flux.
"""
@parallel function update_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz) * _β_dτ
    return nothing
end

"""
Helper function which:
    1) Computes Darcy flux using `compute_flux_p_3D!()`
    2) Updates pressure accordingly using `update_Pf_3D!()`
"""
function compute_pressure_3D!(
    Pf, T, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ, k_ηf, _1_θ_dτ, αρg
)
    @parallel compute_flux_p_3D!(qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ, αρg)
    @parallel update_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    return nothing
end

"""
Compute pressure fluxes and gradients.
"""
@parallel function compute_flux_T_3D!(
    T, qTx, qTy, qTz, gradTx, gradTy, gradTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
)
    @all(qTx) = @all(qTx) - (@all(qTx) + λ_ρCp * _dx * @d_xi(T)) * _1_θ_dτ_T
    @all(qTy) = @all(qTy) - (@all(qTy) + λ_ρCp * _dy * @d_yi(T)) * _1_θ_dτ_T
    @all(qTz) = @all(qTz) - (@all(qTz) + λ_ρCp * _dz * @d_zi(T)) * _1_θ_dτ_T
    @all(gradTx) = @d_xi(T) * _dx
    @all(gradTy) = @d_yi(T) * _dy
    @all(gradTz) = @d_zi(T) * _dz
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
    @parallel compute_flux_T_3D!(
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
