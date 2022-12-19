using Plots
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
function plot_perf(T_effs_parallel, T_effs_inidices, ns)
    T_peak = 537
    T_adv = 732
    hline([T_peak]; ls=:dash, color=:gray, label="Peak measured")
    hline!([T_adv]; ls=:dot, color=:gray, label="Peak advertised")
    plot!(
        ns,
        T_effs_parallel;
        xlab="Problem size (nx=ny)",
        ylab="Memory Bandwidth [GB/s]",
        label="@parallel",
        title="Diffusion Solver on P100 GPU using ParallelStencil",
    )
    p = plot!(ns, T_effs_indices; label="@parallel_indices")
    return savefig(p, "../docs/parallel_vs_indices.png")
end

ns = 32 .* 2 .^ (0:8) .- 1
T_effs_parallel = [
    2.6571428571428575,
    10.922600619195048,
    42.181104936253675,
    154.92901816737816,
    238.9166809630011,
    307.8064191963137,
    307.4926685185837,
    310.60256836212216,
    340.78370708787435,
]
T_effs_indices = [
    2.6319753509072235,
    10.857240553940846,
    42.46569030771763,
    154.25521399624395,
    238.92123522683949,
    308.6783706225382,
    335.8357772827913,
    310.0751387302498,
    340.5800459149292,
]

plot_perf(T_effs_parallel, T_effs_indices, ns)
