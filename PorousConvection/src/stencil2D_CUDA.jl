module stencil2D_CUDA
export compute_flux_p_2D!,
    compute_Pf_2D!,
    compute_pressure_2D!,
    compute_flux_T_2D!,
    computedTdt_2D!,
    update_T_2D!,
    bc_x!,
    compute_temp_2D!

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2)

@parallel function compute_flux_p_2D!(qDx, qDy, Pf, T, k_ηf, _dx, _dy, αρgx, αρgy, _1_θ_dτ)
    @inn_x(qDx) =
        @inn_x(qDx) - (@inn_x(qDx) + k_ηf * @d_xa(Pf) * _dx - αρgx * @av_xa(T)) * _1_θ_dτ
    @inn_y(qDy) =
        @inn_y(qDy) - (@inn_y(qDy) + k_ηf * @d_ya(Pf) * _dy - αρgy * @av_ya(T)) * _1_θ_dτ
    return nothing
end
@parallel function compute_Pf_2D!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy) * _β_dτ
    return nothing
end

function compute_pressure_2D!(Pf, T, qDx, qDy, _dx, _dy, _β_dτ, k_ηf, αρgx, αρgy, _1_θ_dτ)
    @parallel compute_flux_p_2D!(qDx, qDy, Pf, T, k_ηf, _dx, _dy, αρgx, αρgy, _1_θ_dτ)
    @parallel compute_Pf_2D!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
    return nothing
end
@parallel function compute_flux_T_2D!(
    T, qTx, qTy, gradTx, gradTy, λ_ρCp, _dx, _dy, _1_θ_dτ_T
)
    @all(qTx) = @all(qTx) - (@all(qTx) + λ_ρCp * _dx * @d_xi(T)) * _1_θ_dτ_T
    @all(qTy) = @all(qTy) - (@all(qTy) + λ_ρCp * _dy * @d_yi(T)) * _1_θ_dτ_T
    @all(gradTx) = @d_xi(T) * _dx
    @all(gradTy) = @d_yi(T) * _dx
    return nothing
end

@parallel_indices (ix, iy) function computedTdt_2D!(
    dTdt, T, T_old, gradTx, gradTy, qDx, qDy, _dt, _ϕ
)
    nx, ny = size(dTdt)
    if (ix <= nx && iy <= ny)
        dTdt[ix, iy] =
            (T[ix + 1, iy + 1] - T_old[ix + 1, iy + 1]) * _dt +
            (
                max(0.0, qDx[ix + 1, iy + 1]) * gradTx[ix, iy] +
                min(0.0, qDx[ix + 2, iy + 1]) * gradTx[ix + 1, iy] +
                max(0.0, qDy[ix + 1, iy + 1]) * gradTy[ix, iy] +
                min(0.0, qDy[ix + 1, iy + 2]) * gradTy[ix, iy + 1]
            ) * _ϕ
    end
    return nothing
end

@parallel function update_T_2D!(T, dTdt, qTx, qTy, _dx, _dy, _β_dt)
    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy) * _β_dt
    return nothing
end

@parallel_indices (iy) function bc_x!(A)
    A[1, iy] = A[2, iy]
    A[end, iy] = A[end - 1, iy]
    return nothing
end

function compute_temp_2D!(
    T,
    T_old,
    dTdt,
    qTx,
    qTy,
    gradTx,
    gradTy,
    qDx,
    qDy,
    _dx,
    _dy,
    _dt,
    _1_dt_β_dτ_T,
    λ_ρCp,
    _1_θ_dτ_T,
    _ϕ,
)
    @parallel compute_flux_T_2D!(T, qTx, qTy, gradTx, gradTy, λ_ρCp, _dx, _dy, _1_θ_dτ_T)
    @parallel computedTdt_2D!(dTdt, T, T_old, gradTx, gradTy, qDx, qDy, _dt, _ϕ)
    @parallel update_T_2D!(T, dTdt, qTx, qTy, _dx, _dy, _1_dt_β_dτ_T)
    @parallel (1:size(T, 2)) bc_x!(T)
    return nothing
end
end
