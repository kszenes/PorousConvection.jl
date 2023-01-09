##
using Plots, Plots.Measures, Printf
default(
    size = (1200, 800),
    framestyle = :box,
    label = false,
    grid = false,
    margin = 10mm,
    lw = 6,
    labelfontsize = 20,
    tickfontsize = 20,
    titlefontsize = 24,
)
# overcome bug "No current plot/subplot"
plot()
##
@views function porous_convection_2D(debug = false)
    # physics
    lx, ly = 40.0, 20.0
    k_ηf = 1.0
    αρgx, αρgy = 0.0, 1.0
    αρg = sqrt(αρgx^2 + αρgy^2)
    ΔT = 200.0
    ϕ = 0.1
    Ra = 1000.0
    λ_ρCp = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    nvis = 5
    nt = 500
    nx = 127
    ny = ceil(Int, nx * ly / lx)
    ϵtol = 1e-8
    maxiter = 100max(nx, ny)
    ncheck = ceil(Int, 0.25max(nx, ny))
    re = 2π
    cfl = 1 / sqrt(2.1)
    # derived numerics
    dx, dy = lx / nx, ly / ny
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly + dy / 2, -dy / 2, ny)
    θ_dτ = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    dt_diff = min(dx, dy)^2 / λ_ρCp / 4.1
    # array initialisation
    #   pressure 
    Pf = zeros(nx, ny)
    qDx = zeros(nx + 1, ny)
    qDy = zeros(nx, ny + 1)

    #   temperature
    T = @. ΔT * exp(-xc^2 - (yc' + ly / 2)^2)
    T[:, 1] .= ΔT / 2
    T[:, end] .= -ΔT / 2
    T[[1, end], :] .= T[[2, end - 1], :]
    qTx = zeros(nx - 1, ny - 2)
    qTy = zeros(nx - 2, ny - 1)
    gradTx = zeros(nx - 1, ny - 2)
    gradTy = zeros(nx - 2, ny - 1)
    # visualisation init
    st = ceil(Int, nx / 25)
    Xc, Yc = [x for x in xc, y in yc], [y for x in xc, y in yc]
    Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]
    qDxc = zeros(nx, ny)
    qDyc = zeros(nx, ny)
    qDx_p = zeros(Int(ceil(nx / st)), Int(ceil(ny / st)))
    qDy_p = zeros(Int(ceil(nx / st)), Int(ceil(ny / st)))
    qDmag = zeros(nx, ny)
    # time loop
    anim = @animate for it = 1:nt
        iter = 1
        err_Pf = 2ϵtol
        # --- Pressure ---
        # iteration loop
        while err_Pf >= ϵtol && iter <= maxiter
            qDx[2:end-1, :] .-=
                (
                    qDx[2:end-1, :] .+
                    k_ηf .* (
                        diff(Pf, dims = 1) ./ dx .-
                        αρgx .* (T[1:end-1, :] .+ T[2:end, :]) ./ 2
                    )
                ) ./ (θ_dτ .+ 1.0)
            qDy[:, 2:end-1] .-=
                (
                    qDy[:, 2:end-1] .+
                    k_ηf .* (
                        diff(Pf, dims = 2) ./ dy .-
                        αρgy .* (T[:, 1:end-1] .+ T[:, 2:end]) ./ 2
                    )
                ) ./ (θ_dτ .+ 1.0)
            Pf .-= (diff(qDx, dims = 1) ./ dx + diff(qDy, dims = 2) ./ dy) ./ β_dτ
            if iter % ncheck == 0
                err_Pf =
                    maximum(abs.(diff(qDx, dims = 1) ./ dx + diff(qDy, dims = 2) ./ dy))
                if debug
                    @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
                end
            end
            iter += 1
        end
        # --- Temperature ---
        dt_adv = ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1
        dt = min(dt_diff, dt_adv)
        qTx .= diff(T[:, 2:end-1], dims = 1) ./ dx
        qTy .= diff(T[2:end-1, :], dims = 2) ./ dy
        # diffusion
        T[2:end-1, 2:end-1] .+=
            dt .* λ_ρCp .* (diff(qTx, dims = 1) ./ dx .+ diff(qTy, dims = 2) ./ dy)
        # upwinding (advection)
        gradTx .= diff(T[:, 2:end-1], dims = 1) ./ dx
        gradTy .= diff(T[2:end-1, :], dims = 2) ./ dy
        T[2:end-1, 2:end-1] .-=
            dt ./ ϕ .* (
                max.(0.0, qDx[2:end-1, 2:end-1][1:end-1, :]) .* gradTx[1:end-1, :] .+
                min.(0.0, qDx[2:end-1, 2:end-1][2:end, :]) .* gradTx[2:end, :] .+
                max.(0.0, qDy[2:end-1, 2:end-1][:, 1:end-1]) .* gradTy[:, 1:end-1] .+
                min.(0.0, qDy[2:end-1, 2:end-1][:, 2:end]) .* gradTy[:, 2:end]
            )
        # adiabatic boundary conditions
        T[[1, end], :] .= T[[2, end - 1], :]
        # output
        if it % nvis == 0
            @printf("it = %d, iter/nx=%.1f, err_Pf=%1.3e\n", it, iter / nx, err_Pf)
            qDxc .= (qDx[1:end-1, :] .+ qDx[2:end, :]) ./ 2
            qDyc .= (qDy[:, 1:end-1] .+ qDy[:, 2:end]) ./ 2
            qDmag .= sqrt.(qDxc .^ 2 .+ qDyc .^ 2)
            qDxc ./= qDmag
            qDyc ./= qDmag
            qDx_p .= qDxc[1:st:end, 1:st:end]
            qDy_p .= qDyc[1:st:end, 1:st:end]
            heatmap(
                xc,
                yc,
                T';
                xlims = (xc[1], xc[end]),
                ylims = (yc[1], yc[end]),
                title = "Temperature",
                aspect_ratio = 1,
                c = :turbo,
            )
            quiver!(Xp[:], Yp[:], quiver = (qDx_p[:], qDy_p[:]), lw = 0.5, c = :black)
        end
    end
    gif(anim, "img/porous_convection_2D.gif", fps = 30)
end
##
porous_convection_2D()
##
