using Plots

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid = open(fname, "r")
    read!(fid, A)
    return close(fid)
end

function visualise(nx, ny, nz)
    lx, ly, lz = 40.0, 20.0, 20.0
    T = zeros(Float32, nx, ny, nz)
    load_array("out_T", T)
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    _dx, _dy, _dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(dy / 2, ly - dy / 2, ny)
    zc = LinRange(-ly + dy / 2, -dy / 2, ny)
    p = Plots.heatmap(
        xc,
        zc,
        T[:, ceil(Int, ny / 2), :]';
        c=:turbo,
        aspect_ratio=1,
        xlabel="lx",
        ylabel="lz",
        title="Temperature",
    )
    return savefig(p, "T_slice.png")
end

visualise(255, 127, 127)
