using GLMakie
using Plots

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid = open(fname, "r")
    read!(fid, A)
    return close(fid)
end

function visualise(nx, ny, nz, dir="viz3Dmpi_out/")
    lx, ly, lz = 40.0, 20.0, 20.0
    T = zeros(Float32, nx, ny, nz)
    T_plot = Observable(T)
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    _dx, _dy, _dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(dy / 2, ly - dy / 2, ny)
    zc = LinRange(-ly + dy / 2, -dy / 2, ny)
    fig = Figure(; resolution=(1600, 1000), fontsize=24)
    ax = Axis3(
        fig[1, 1];
        aspect=(1, 1, 0.5),
        title="Temperature",
        xlabel="lx",
        ylabel="ly",
        zlabel="lz",
    )
    for (root, dirs, files) in walkdir(dir)
        frames = 1:size(files)[1]
        record(fig, "../docs/porous-3d-multixpu.gif", frames; framerate=15) do i
            file = files[i]
            file = splitext(file)[1]
            load_array(dir * file, T)
            T_plot[] = T
            surf_T = GLMakie.contour!(ax, xc, yc, zc, T_plot; alpha=0.05, colormap=:turbo)
        end
    end
end

visualise(506, 250, 250)
