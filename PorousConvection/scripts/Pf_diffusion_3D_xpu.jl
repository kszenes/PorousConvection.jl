const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
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

@parallel function compute_flux!(qDx, qDy, qDz, Pf, k_ηf, _dx, _dy, _dz, _1_θ_dτ)
    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf * _dx * @d_xa(Pf)) * _1_θ_dτ
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf * _dy * @d_ya(Pf)) * _1_θ_dτ
    @inn_z(qDz) = @inn_z(qDz) - (@inn_z(qDz) + k_ηf * _dz * @d_za(Pf)) * _1_θ_dτ
    return nothing
end
@parallel function compute_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    @all(Pf) = @all(Pf) - (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy + @d_za(qDz) * _dz) * _β_dτ
    return nothing
end

function compute!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ, k_ηf, _1_θ_dτ)
    @parallel compute_flux!(qDx, qDy, qDz, Pf, k_ηf, _dx, _dy, _dz, _1_θ_dτ)
    @parallel compute_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ)
    return nothing
end

function Pf_diffusion_2D_perf_loop_fun(; n=511, do_check=false)
    # physics
    lx, ly, lz = 20.0, 20.0, 20
    k_ηf = 1.0
    # numerics
    nx = ny = nz = n
    ϵtol = 1e-8
    maxiter = 50
    cfl = 1.0 / sqrt(3.1)
    re = 2π
    ncheck = 1
    # derived numerics
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    _dx, _dy, _dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    xc = LinRange(dx / 2, lx - dx / 2, nx)
    yc = LinRange(dy / 2, ly - dy / 2, ny)
    zc = LinRange(dz / 2, lz - dz / 2, nz)
    θ_dτ = max(lx, ly, lz) / re / cfl / min(dx, dy, dz)
    β_dτ = (re * k_ηf) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
    _β_dτ = 1.0 ./ β_dτ
    _1_θ_dτ = 1.0 ./ (1.0 + θ_dτ)
    # array initialisation
    Pf = Data.Array([
        exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for
        ix in 1:nx, iy in 1:ny, iz in 1:nz
    ])
    display(
        heatmap(
            xc,
            zc,
            Array(Pf)[:, ceil(Int, ny / 2), :]';
            xlims=(xc[1], xc[end]),
            ylims=(zc[1], zc[end]),
            aspect_ratio=1,
            c=:turbo,
        ),
    )
    sleep(10)
    qDx = @zeros(nx + 1, ny, nz)
    qDy = @zeros(nx, ny + 1, nz)
    qDz = @zeros(nx, ny, nz + 1)
    # iteration loop
    iter = 1
    err_Pf = 2ϵtol
    # timing
    t_tic = 0.0
    niter = 0
    while err_Pf >= ϵtol && iter <= maxiter
        # ignore warmup
        if iter == 11
            t_tic = Base.time()
            niter = 0
        end
        compute!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ, k_ηf, _1_θ_dτ)
        if do_check && iter % ncheck == 0
            err_Pf = maximum(
                abs.(
                    diff(qDx; dims=1) .* _dx .+ diff(qDy; dims=2) .* _dy .+
                    diff(qDz; dims=3)
                ),
            )
            @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
            display(
                heatmap(
                    xc,
                    zc,
                    Array(Pf)[:, ceil(Int, ny / 2), :]';
                    xlims=(xc[1], xc[end]),
                    ylims=(zc[1], zc[end]),
                    aspect_ratio=1,
                    c=:turbo,
                ),
            )
            # display(
            #     heatmap(
            #         xc,
            #         yc,
            #         Array(Pf');
            #         xlims = (xc[1], xc[end]),
            #         ylims = (yc[1], yc[end]),
            #         aspect_ratio = 1,
            #         c = :turbo,
            #     ),
            # )
        end
        iter += 1
        niter += 1
    end
    t_toc_man = Base.time() - t_tic
    t_toc_bench = @belapsed begin
        compute!($Pf, $qDx, $qDy, $qDz, $_dx, $_dy, $_dz, $_β_dτ, $k_ηf, $_1_θ_dτ)
    end
    A_eff = 6 * sizeof(eltype(Pf)) * (nx * ny * nz) * 1e-9
    t_it_man = t_toc_man / niter
    T_eff_man = A_eff / t_it_man
    @printf(
        "Manual timing: Time = %1.3f [s], T_eff = %1.3f [GB/s]; iter = %d \n",
        t_toc_man,
        T_eff_man,
        niter
    )
    niter = 1
    t_it_bench = t_toc_bench / niter
    T_eff_bench = A_eff / t_it_bench
    @printf(
        "@belapsed:  Time = %1.3f [s], T_eff = %1.3f [GB/s]; iter = %d \n\n",
        t_toc_bench,
        T_eff_bench,
        niter
    )
end

Pf_diffusion_2D_perf_loop_fun(; n=40, do_check=true)

function Pf_diffusion_benchmark(; n=127, do_check=false)
    # physics
    lx, ly = 20.0, 20.0
    k_ηf = 1.0
    # numerics
    nx = ny = n
    cfl = 1.0 / sqrt(2.1)
    re = 2π
    ncheck = 1
    # derived numerics
    dx, dy = lx / nx, ly / ny
    xc, yc = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    θ_dτ = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    _β_dτ = 1.0 ./ β_dτ
    _dx, _dy = 1.0 / dx, 1.0 / dy
    k_ηf_dx, k_ηf_dy = k_ηf / dx, k_ηf / dy
    _1_θ_dτ = 1.0 ./ (1.0 + θ_dτ)
    # array initialisation
    Pf = Data.Array(@. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2))
    qDx = @zeros(nx + 1, ny)
    qDy = @zeros(nx, ny + 1)
    # iteration loop
    t_toc_bench = @belapsed begin
        compute!($Pf, $qDx, $qDy, $qDz, $_dx, $_dy, $_dz, $_β_dτ, $k_ηf, $_1_θ_dτ)
    end
    A_eff = 6 * sizeof(eltype(Pf)) * (nx * ny, nz) * 1e-9
    niter = 1
    t_it_bench = t_toc_bench / niter
    T_eff_bench = A_eff / t_it_bench
    @printf(
        "n = %d:  Time = %.3f [s], T_eff = %1.3f [GB/s]%d\n",
        n,
        t_toc_bench,
        T_eff_bench,
        niter
    )
    return T_eff_bench
end

function run_benchmark()
    T_peak = 537
    T_adv = 732
    ns = 32 .* 2 .^ (0:8) .- 1
    T_effs = []
    for n in ns
        push!(T_effs, Pf_diffusion_benchmark(; n=n, do_check=false))
    end
    hline([T_peak]; ls=:dash, color=:gray, label="Peak measured")
    hline!([T_adv]; ls=:dot, color=:gray, label="Peak advertised")
    p = plot!(
        ns,
        T_effs;
        xlab="Problem size (nx=ny)",
        ylab="Memory Bandwidth [GB/s]",
        label="Solver",
        title="Diffusion Solver on P100 GPU using @parallel",
    )
    return savefig(p, "../docs/diffusion_parallel.png")
end
run_benchmark()
