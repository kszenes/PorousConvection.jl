using Printf

@views function porous_convection_implicit_2D(;
    Ra=1000.0,
    nt=4000,
    nx=511,
    ny=1023,
    nvis=50,
    fname="porous_long",
    debug=false,
    save=true,
)
    # physics
    lx, ly = 40.0, 20.0
    k_ηf = 1.0
    αρgx, αρgy = 0.0, 1.0
    αρg = sqrt(αρgx^2 + αρgy^2)
    ΔT = 200.0
    ϕ = 0.1
    _ϕ = 1.0 / ϕ
    λ_ρCp = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    ϵtol = 1e-6
    maxiter = 10max(nx, ny)
    ncheck = ceil(2max(nx, ny))
    re_D = 4π
    cfl = 1 / sqrt(2.1)
    # derived numerics
    dx, dy = lx / nx, ly / ny
    _dx, _dy = 1.0 / dx, 1.0 / dy
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly + dy / 2, -dy / 2, ny)
    θ_dτ_D = max(lx, ly) / re_D / cfl / min(dx, dy)
    _1_θ_dτ_D = 1.0 / (1.0 + θ_dτ_D)
    _β_dτ_D = 1.0 / ((re_D * k_ηf) / (cfl * min(dx, dy) * max(lx, ly)))
    # array initialisation
    #  pressure
    Pf = @zeros(nx, ny)
    qDx = @zeros(nx + 1, ny)
    qDy = @zeros(nx, ny + 1)

    #  temperature
    T = @. ΔT * exp(-xc^2 - (yc' + ly / 2)^2)
    T[:, 1] .= ΔT / 2
    T[:, end] .= -ΔT / 2
    T[[1, end], :] .= T[[2, end - 1], :]
    T = Data.Array(T)
    T_old = @zeros(nx, ny)
    qTx = @zeros(nx - 1, ny - 2)
    qTy = @zeros(nx - 2, ny - 1)
    dTdt = @zeros(nx - 2, ny - 2)
    r_Pf = zeros(nx, ny)
    r_T = zeros(nx - 2, ny - 2)
    gradTx = @zeros(nx - 1, ny - 2)
    gradTy = @zeros(nx - 2, ny - 1)
    # visualisation init
    st = ceil(Int, nx / 25)
    Xc, Yc = [x for x in xc, y in yc], [y for x in xc, y in yc]
    Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]
    qDxc = zeros(nx, ny)
    qDyc = zeros(nx, ny)
    qDmag = zeros(nx, ny)
    # time loop
    for it in 1:nt
        T_old .= T
        # time step
        dt = if it == 1
            0.1 * min(dx, dy) / (αρg * ΔT * k_ηf)
        else
            min(
                5.0 * min(dx, dy) / (αρg * ΔT * k_ηf),
                ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1,
            )
        end
        _dt = 1.0 / dt
        # implicit temperature params
        re_T = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
        θ_dτ_T = max(lx, ly) / re_T / cfl / min(dx, dy)
        _1_θ_dτ_T = 1.0 / (1.0 + θ_dτ_T)
        β_dτ_T = ((re_T * λ_ρCp) / (cfl * min(dx, dy) * max(lx, ly)))
        _1_dt_β_dτ_T = 1.0 / (1.0 / dt + β_dτ_T)

        iter = 1
        err_D = 2ϵtol
        err_T = 2ϵtol
        # iteration loop
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # --- Pressure ---
            compute_pressure_2D!(
                Pf, T, qDx, qDy, _dx, _dy, _β_dτ_D, k_ηf, αρgx, αρgy, _1_θ_dτ_D
            )
            # --- Temperature ---
            compute_temp_2D!(
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
            if iter % ncheck == 0
                r_Pf .= Array(diff(qDx; dims=1) ./ dx + diff(qDy; dims=2) ./ dy)
                r_T .= Array(dTdt .+ diff(qTx; dims=1) ./ dx + diff(qTy; dims=2) ./ dy)
                err_D = maximum(abs.(r_Pf))
                err_T = maximum(abs.(r_T))

                if debug
                    @printf(
                        "  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n",
                        iter / nx,
                        err_D,
                        err_T
                    )
                end
            end
            iter += 1
        end
        # output
        if it % nvis == 0
            @printf(
                "it = %d, iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n",
                it,
                iter / nx,
                err_D,
                err_T
            )
            qDxc .= Array((qDx[1:(end - 1), :] .+ qDx[2:end, :]) ./ 2)
            qDyc .= Array((qDy[:, 1:(end - 1)] .+ qDy[:, 2:end]) ./ 2)
            qDmag .= sqrt.(qDxc .^ 2 .+ qDyc .^ 2)
            qDxc ./= qDmag
            qDyc ./= qDmag
            qDx_p = qDxc[1:st:end, 1:st:end]
            qDy_p = qDyc[1:st:end, 1:st:end]
            heatmap(
                xc,
                yc,
                Array(T');
                xlims=(xc[1], xc[end]),
                ylims=(yc[1], yc[end]),
                title="Temperature",
                aspect_ratio=1,
                c=:turbo,
            )
            p = quiver!(Xp[:], Yp[:]; quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black)
            if save
                savefig(p, "viz_out/$(fname)_$(it).png")
            end
        end
    end
    return Array(T)
end
