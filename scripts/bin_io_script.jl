using Plots
"""
Writes array to .bin file
"""
function save_array(Aname, A)
    fname = string(Aname, ".bin")
    out = open(fname, "w")
    write(out, A)
    return close(out)
end

"""
Loads array from .bin file
"""
function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid = open(fname, "r")
    read!(fid, A)
    return close(fid)
end

function main()
    A = rand(3, 3)
    B = zeros(3, 3)

    save_array("A", A)
    load_array("A", B)
    return B
end

B = main()
heatmap(B)
