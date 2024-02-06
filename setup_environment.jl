if length(ARGS)>0
    if ARGS[1] == "cluster"
        using Libdl
        push!(Libdl.DL_LOAD_PATH, ENV["HDF5_DIR"] * "/lib")
    end
end

using Pkg
Pkg.add(url = "https://github.com/axelparmentier/InferOpt.jl.git", rev = "generalized-maximizer")
Pkg.add("ProgressMeter")
Pkg.add("Flux")
Pkg.add("Gurobi")
Pkg.add("JuMP")
Pkg.add("Distributions")
Pkg.add("Statistics")
Pkg.add("JSON")
Pkg.add("Dates")
Pkg.add("Parameters")
Pkg.add("BSON")
Pkg.add("ArgParse")