


module GVFN
using Flux
using Reexport

@reexport using JuliaRL

include("Environments.jl")

# export GVFNetwork
# include("GVFNetwork.jl")

export jacobian, glorot_uniform, glorot_normal
include("util.jl")


export
    GVF,
    get,
    cumulant,
    discount,
    policy,
    Horde,
    NullPolicy,
    ConstantDiscount,
    StateTerminationDiscount,
    FeatureCumulant,
    PredictionCumulant



include("GVF.jl")

export GVFNetwork, reset!, get
include("GVFNetwork.jl")


export RTD, RTD_jacobian, TDλ, train!
include("Update.jl")








end
