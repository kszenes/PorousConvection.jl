using ImplicitGlobalGrid
using Printf
import MPI
MPI.Init()

max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

function save_array(Aname, A)
    fname = string(Aname, ".bin")
    out = open(fname, "w")
    write(out, A)
    return close(out)
end

function porous_convection_implicit_3D(;
    Ra=1000.0, nt=500, nz=63, nvis=50, debug=false, save=true
)
    # physics
    lx, ly, lz = 40.0, 20.0, 20.0
    k_ηf = 1.0
    αρg = 1.0
    ΔT = 200.0
    ϕ = 0.1
    _ϕ = 1.0 / ϕ
    λ_ρCp = 1 / Ra * (αρg * k_ηf * ΔT * lz / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    ϵtol = 1e-6
    nx, ny = 2 * (nz + 1) - 1, nz
    me, dims = init_global_grid(nx, ny, nz, init_MPI=false)  # init global grid and more
    # maxiter = 10
    maxiter = 10max(nx_g(), ny_g())
    b_width = (8, 8, 4)                       # for comm / comp overlap
    cfl = 1.0 / sqrt(3.1)
    re_D = 4π
    ncheck = ceil(2max(nx_g(), ny_g(), nz_g()))
    # derived numerics
    dx = lx / nx_g()
    dy = ly / ny_g()
    dz = lz / nz_g()
    _dx, _dy, _dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_g())
    yc = LinRange(dy / 2, ly - dy / 2, ny_g())
    zc = LinRange(-ly + dy / 2, -dy / 2, ny_g())
    θ_dτ_D = max(lx, ly, lz) / re_D / cfl / min(dx, dy, dz)
    β_dτ_D = (re_D * k_ηf) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
    _β_dτ_D = 1.0 ./ β_dτ_D
    _1_θ_dτ_D = 1.0 ./ (1.0 + θ_dτ_D)
    # visualisation setup
    if save
        ENV["GKSwstype"] = "nul"
        if (me == 0)
            if isdir("viz3Dmpi_out") == false
                mkdir("viz3Dmpi_out")
            end
            loadpath = "viz3Dmpi_out/"
            anim = Animation(loadpath, String[])
            println("Animation directory: $(anim.dir)")
        end
        nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]
        if (nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory())
            error("Not enough memory for visualization.")
        end
        T_v = zeros(nx_v, ny_v, nz_v) # global array for visu
        T_inn = zeros(nx - 2, ny - 2, nz - 2) # no halo local array for visu
        xi_g, zi_g = LinRange(-lx / 2 + dx + dx / 2, lx / 2 - dx - dx / 2, nx_v),
        LinRange(-lz + dz + dz / 2, -dz - dz / 2, nz_v) # inner points only
        iframe = 0
    end
    # --- array initialisation ---
    # Pressure
    Pf = @zeros(nx, ny, nz)
    qDx = @zeros(nx + 1, ny, nz)
    qDy = @zeros(nx, ny + 1, nz)
    qDz = @zeros(nx, ny, nz + 1)
    # Temperature
    T = @zeros(nx, ny, nz)
    T .= Data.Array([
        ΔT * exp(
            -(x_g(ix, dx, T) + dx / 2 - lx / 2)^2 - (y_g(iy, dy, T) + dy / 2 - ly / 2)^2 -
            (z_g(iz, dz, T) + dz / 2 - lz / 2)^2,
        ) for ix in 1:size(T, 1), iy in 1:size(T, 2), iz in 1:size(T, 3)
    ])
    T[:, :, 1] .= ΔT / 2
    T[:, :, end] .= -ΔT / 2
    update_halo!(T)
    T_old = copy(T)
    qTx = @zeros(nx - 1, ny - 2, nz - 2)
    qTy = @zeros(nx - 2, ny - 1, nz - 2)
    qTz = @zeros(nx - 2, ny - 2, nz - 1)
    dTdt = @zeros(nx - 2, ny - 2, nz - 2)
    r_Pf = zeros(nx, ny, nz)
    r_T = zeros(nx - 2, ny - 2, nz - 2)

    threads = (32, 4, 4)
    blocks  = (size(Pf) .+ threads .- 1) .÷ threads

    # Disable garbace collection for accurate benchmarking
    GC.gc(); GC.enable(false)
    t_tic = 0.0
    t_toc = 0.0
    niter = 0
    # Time loop
    for it in 1:nt
        T_old .= T
        # time step
        dt = if it == 1
            0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
        else
            min(
                5.0 * min(dx, dy, dz) / (αρg * ΔT * k_ηf),
                ϕ * min(
                    dx / max_g(abs.(qDx)), dy / max_g(abs.(qDy)), dz / max_g(abs.(qDz))
                ) / 3.1,
            )
        end
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
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            if (iter == 4)
                t_tic = Base.time()
            end
            # --- Pressure ---
            # @hide_communication b_width begin
                @parallel blocks threads shmem=(prod(threads.+1))*sizeof(eltype(Pf)) compute_flux_p_3D!(qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ_D, αρg)
                update_halo!(qDx, qDy, qDz)
                # @parallel compute_flux_p_3D!(
                #     qDx, qDy, qDz, Pf, T, k_ηf, _dx, _dy, _dz, _1_θ_dτ_D, αρg
                # )
            # end

            # @hide_communication b_width begin
                # @parallel blocks threads compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)
                @parallel blocks threads compute_Pf_3D!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)
                update_halo!(Pf)
            # end

            
            @parallel blocks threads shmem=prod(threads.+1)*sizeof(eltype(T)) compute_flux_T_3D!(
                T, qTx, qTy, qTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
            )
            # @parallel compute_flux_T_3D!(
            #     T, qTx, qTy, qTz, gradTx, gradTy, gradTz, λ_ρCp, _dx, _dy, _dz, _1_θ_dτ_T
            # )

            # @parallel computedTdt_3D!(
            #     dTdt, T, T_old, gradTx, gradTy, gradTz, qDx, qDy, qDz, _dt, _ϕ
            # )
            # --- Temperature ---
            # @hide_communication b_width begin
                @parallel blocks threads shmem=prod(threads.+2)*sizeof(eltype(T)) computedTdt_3D!(
                    dTdt, T, T_old, qDx, qDy, qDz, _dx, _dy, _dz, _dt, _ϕ
                )
                @parallel blocks threads update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _1_dt_β_dτ_T)
                # @parallel update_T_3D!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _1_dt_β_dτ_T)
                @parallel (1:size(T, 2), 1:size(T, 3)) bc_xz!(T)
                @parallel (1:size(T, 1), 1:size(T, 3)) bc_yz!(T)
                update_halo!(T)
            # end
            niter += 1
            if iter % ncheck == 0
                # Don't include error computation in timing
                t_toc += Base.time() - t_tic

                r_Pf .= Array(
                    diff(qDx; dims=1) ./ dx .+ diff(qDy; dims=2) ./ dy .+
                    diff(qDz; dims=3) ./ dz,
                )
                r_T .= Array(
                    dTdt .+ diff(qTx; dims=1) ./ dx .+ diff(qTy; dims=2) ./ dy .+
                    diff(qTz; dims=3) ./ dz,
                )
                err_D = max_g(abs.(r_Pf))
                err_T = max_g(abs.(r_T))

                if debug && me == 0
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
        # visualisation
        if save && (it % nvis == 0)
            T_inn .= Array(T)[2:(end - 1), 2:(end - 1), 2:(end - 1)]
            gather!(T_inn, T_v)
            if me == 0
                @printf(
                    "it = %d, iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n",
                    it,
                    iter / nx,
                    err_D,
                    err_T
                )
                p1 = heatmap(
                    xi_g,
                    zi_g,
                    T_v[:, ceil(Int, ny_g() / 2), :]';
                    xlims=(xi_g[1], xi_g[end]),
                    ylims=(zi_g[1], zi_g[end]),
                    aspect_ratio=1,
                    c=:turbo,
                )
                # display(p1)
                png(p1, @sprintf("viz3Dmpi_out/%04d.png", iframe += 1))
                save_array(
                    @sprintf("viz3Dmpi_out/out_T_%04d", iframe), convert.(Float32, T_v)
                )
            end
        end
    end
    A_eff = 32 * nx_g() * ny_g() * nz_g() * sizeof(Float64) * 1e-9
    t_it = t_toc / niter
    T_eff = A_eff / t_it
    
    
    @show t_it, niter
    @show T_eff
    GC.enable(true);
    finalize_global_grid(; finalize_MPI=false)
    return T_eff, t_it
end
