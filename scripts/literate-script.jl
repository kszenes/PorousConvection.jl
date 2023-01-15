using Literate
ENV["GKSwstype"] = "nul"
Literate.markdown("bin_io_script.jl", "./"; execute=true, documenter=false, credit=false)
