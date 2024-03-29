push!(LOAD_PATH, "../src")

using PorousConvection

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()

    printstyled("Testing PorousConvection.jl\n"; bold=true, color=:white)

    return run(
        `$exename -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, "test3D.jl"))`
    )
end
runtests()
exit()
