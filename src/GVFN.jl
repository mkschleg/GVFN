
__precompile__(true)

module GVFN
using Flux
using Reexport

@reexport using JuliaRL

include("Environments.jl")

export jacobian, glorot_uniform, glorot_normal, StopGradient
include("util.jl")

export
    SingleLayer,
    Linear,
    deriv,
    sigmoid,
    sigmoid′

include("Layers.jl")

export
    GVF,
    # get, get!,
    cumulant,
    discount,
    policy,
    Horde,
    NullPolicy,
    PersistentPolicy,
    ConstantDiscount,
    StateTerminationDiscount,
    FeatureCumulant,
    PredictionCumulant

include("GVF.jl")


export GVFNetwork, GVFActionNetwork, reset!, get
include("GVFNetwork.jl")

export RTD, RTD_jacobian, TDLambda, TD, train!
include("Loss.jl")
include("Update.jl")

export OnlineJointTD, OnlineTD_RNN, train_step!
include("RNN.jl")


end
