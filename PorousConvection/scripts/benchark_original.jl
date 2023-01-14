const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    using PorousConvection.stencil3D_CUDA_original
else
    @init_parallel_stencil(Threads, Float64, 3)
    using PorousConvection.stencil3D_Threads
end

using Plots, Plots.Measures, Printf, Test, BenchmarkTools
default(;
    size=(600, 500),
    framestyle=:box,
    label=false,
    grid=false,
    margin=10mm,
    lw=2,
    labelfontsize=11,
    tickfontsize=11,
    titlefontsize=11,
)


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
    @parallel (1:size(T, 2), 1:size(T, 3)) bc_xz!(T)
    @parallel (1:size(T, 1), 1:size(T, 3)) bc_yz!(T)
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
    # for it in 1:nt
    #     T_old .= T
    #     # time step
    #     dt = if it == 1
    #        0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)

          dt = 0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
    #     else
    #         min(
    #             5.0 * min(dx, dy, dz) / (αρg * ΔT * k_ηf),
    #             ϕ * min(
    #                 dx / maximum(abs.(qDx)),
    #                 dy / maximum(abs.(qDy)),
    #                 dz / maximum(abs.(qDz)),
    #             ) / 3.1,
    #         )
    #     end
        _dt = 1.0 / dt
        # implicit temperature params
        re_T = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
        θ_dτ_T = max(lx, ly, lz) / re_T / cfl / min(dx, dy, dz)
        _1_θ_dτ_T = 1.0 / (1.0 + θ_dτ_T)
        β_dτ_T = (re_T * λ_ρCp) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
        _1_dt_β_dτ_T = 1.0 / (1.0 / dt + β_dτ_T)

        iter = 1
        err_D = 2ϵtol
        err_T = 2ϵtol
        err_T = 0.0

        # iteration loop
        # while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # --- Pressure ---
            time = @belapsed begin
                @parallel compute_flux_p_3D!($qDx, $qDy, $qDz, $Pf, $T, $k_ηf, $_dx, $_dy, $_dz, $_1_θ_dτ_D, $αρg)
            end
            @printf("compute_flux_p_3D!: %f [s]\n", time)
            time = @belapsed begin
                @parallel compute_Pf_3D!($Pf, $qDx, $qDy, $qDz, $_dx, $_dy, $_dz, $_β_dτ_D)
            end
            @printf("compute_Pf_3D!: %f [s]\n", time)

            # --- Temperature ---
            time = @belapsed begin
                @parallel compute_flux_T_3D!(
                    $T, $qTx, $qTy, $qTz, $gradTx, $gradTy, $gradTz, $λ_ρCp, $_dx, $_dy, $_dz, $_1_θ_dτ_T
                )
            end
            @printf("compute_flux_T_3D!: %f [s]\n", time)
            time = @belapsed begin
                @parallel computedTdt_3D!(
                    $dTdt, $T, $T_old, $gradTx, $gradTy, $gradTz, $qDx, $qDy, $qDz, $_dt, $_ϕ
                )
            end
            @printf("computedTdt!: %f [s]\n", time)
            time = @belapsed begin
                @parallel update_T_3D!($T, $dTdt, $qTx, $qTy, $qTz, $_dx, $_dy, $_dz, $_1_dt_β_dτ_T)
            end
            @printf("update_T_3D!: %f [s]\n", time)
            time = @belapsed begin
                @parallel (1:size($T, 2), 1:size($T, 3)) bc_xz!($T)
            end
            @printf("bc_xz!: %f [s]\n", time)
            time = @belapsed begin
                @parallel (1:size($T, 1), 1:size($T, 3)) bc_yz!($T)
            end
            @printf("bc_yz!: %f [s]\n", time)
            # if iter % ncheck == 0
            #     r_Pf .= Array(
            #         diff(qDx; dims=1) ./ dx .+ diff(qDy; dims=2) ./ dy .+
            #         diff(qDz; dims=3) ./ dz,
            #     )
            #     r_T .= Array(
            #         dTdt .+ diff(qTx; dims=1) ./ dx .+ diff(qTy; dims=2) ./ dy .+
            #         diff(qTz; dims=3) ./ dz,
            #     )
            #     err_D = maximum(abs.(r_Pf))
            #     err_T = maximum(abs.(r_T))

            #     if debug
            #         @printf(
            #             "  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n",
            #             iter / nx,
            #             err_D,
            #             err_T
            #         )
            #     end
            # end
            iter += 1
        # end
