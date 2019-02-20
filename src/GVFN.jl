


module GVFN
using Reexport

@reexport using JuliaRL

include("Environments.jl")

# export GVFNetwork
# include("GVFNetwork.jl")

export GVFLayer, simple_train
include("GVFLayer.jl")

end
