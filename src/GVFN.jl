
__precompile__(true)

module GVFN
using Flux
using Reexport

@reexport using JuliaRL

import Random

export
    SingleLayer,
    Linear,
    deriv,
    sigmoid,
    sigmoid′,
    relu,
    relu′

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
    PredictionCumulant,
    ScaledCumulant

include("GVF.jl")

export GVFNetwork, GVFActionNetwork, reset!, get
include("GVFNetwork.jl")

export RTD, RTD_jacobian, TDLambda, TD, update!
include("Loss.jl")
include("Update.jl")

export OnlineJointTD, OnlineTD_RNN, train_step!
include("RNN.jl")


include("ActingPolicy.jl")

include("Environments.jl")

export jacobian, glorot_uniform, glorot_normal, StopGradient
include("util.jl")

include("Agent.jl")

end
