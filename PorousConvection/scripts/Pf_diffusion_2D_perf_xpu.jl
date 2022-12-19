const USE_GPU = true
using ParallelStencil
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
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

macro d_xa(A)
    return esc(:($A[ix + 1, iy] - $A[ix, iy]))
end
macro d_ya(A)
    return esc(:($A[ix, iy + 1] - $A[ix, iy]))
end

@parallel_indices (ix, iy) function compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    nx, ny = size(Pf)
    if (ix <= nx - 1 && iy <= ny)
        qDx[ix + 1, iy] =
            qDx[ix + 1, iy] - (qDx[ix + 1, iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ
    end
    if (ix <= nx && iy <= ny - 1)
        qDy[ix, iy + 1] =
            qDy[ix, iy + 1] - (qDy[ix, iy + 1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ
    end
    return nothing
end
@parallel_indices (ix, iy) function compute_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
    nx, ny = size(Pf)
    if (ix <= nx && iy <= ny)
        Pf[ix, iy] = Pf[ix, iy] - ((@d_xa(qDx)) * _dx + (@d_ya(qDy)) * _dy) * _β_dτ
    end
    return nothing
end

function compute!(Pf, qDx, qDy, _dx, _dy, _β_dτ, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    @parallel compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    @parallel compute_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
    return nothing
end

function Pf_diffusion_2D_perf_loop_fun(; n=511, do_check=false)
    # physics
    lx, ly = 20.0, 20.0
    k_ηf = 1.0
    # numerics
    nx = ny = n
    ϵtol = 1e-8
    maxiter = 50
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
        compute!(Pf, qDx, qDy, _dx, _dy, _β_dτ, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        if do_check && iter % ncheck == 0
            err_Pf = maximum(abs.(diff(qDx; dims=1) .* _dx .+ diff(qDy; dims=2) .* _dy))
            @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
            display(
                heatmap(
                    xc,
                    yc,
                    Array(Pf');
                    xlims=(xc[1], xc[end]),
                    ylims=(yc[1], yc[end]),
                    aspect_ratio=1,
                    c=:turbo,
                ),
            )
        end
        iter += 1
        niter += 1
    end
    x_idx = [67, 86, 29, 64, 117]
    y_idx = [103, 119, 96, 21, 88]
    Pf_gpu = Array(Pf[x_idx, y_idx])
    t_toc_man = Base.time() - t_tic
    t_toc_bench = @belapsed begin
        compute!($Pf, $qDx, $qDy, $_dx, $_dy, $_β_dτ, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ)
    end
    A_eff = 6 * sizeof(eltype(Pf)) * (nx * ny) * 1e-9
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
    return Pf_gpu
end

Pf_gpu = Pf_diffusion_2D_perf_loop_fun(; n=127, do_check=false)

Pf_ref = [
    0.05145200526391585 6.712814979475487e-6 0.031741856021777204 0.01709423366560898 -0.013186915837935163
    0.008228472193614148 6.363787926395716e-8 0.05483642202535864 0.001139619275306497 0.036389369428215815
    4.604897326282956e-5 1.5429279015958708e-11 0.00189104848545711 2.622057754369502e-6 0.021714609679119753
    0.05242888559548276 7.284349171481578e-6 0.02987219277174924 0.017780704602589667 -0.01277204233336146
    5.1305341640482385e-12 2.2743645971498444e-20 1.612817554544926e-9 9.454591126515528e-14 1.9256001667330874e-7
]

@test all(isapprox.(Pf_ref, Pf_gpu, atol=1e-16))

function Pf_diffusion_benchmark(; n=127)
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
        compute!($Pf, $qDx, $qDy, $_dx, $_dy, $_β_dτ, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ)
    end
    A_eff = 6 * sizeof(eltype(Pf)) * (nx * ny) * 1e-9
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
        push!(T_effs, Pf_diffusion_benchmark(; n=n))
    end
    hline([T_peak]; ls=:dash, color=:gray, label="Peak measured")
    hline!([T_adv]; ls=:dot, color=:gray, label="Peak advertised")
    p = plot!(
        ns,
        T_effs;
        xlab="Problem size (nx=ny)",
        ylab="Memory Bandwidth [GB/s]",
        label="Solver",
        title="Diffusion Solver on P100 GPU using @parallel_indices",
    )
    savefig(p, "../docs/diffusion_indices.png")
    return T_effs
end
T_eff_indices = run_benchmark()
