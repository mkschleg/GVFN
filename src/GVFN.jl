
__precompile__(true)

module GVFN
import Flux
using Reexport

import Reproduce

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
    ScaledCumulant,
    NormalizedCumulant

include("GVF.jl")

export GVFNetwork, GVFActionNetwork, reset!, get, RNNActionLayer, ForecastNetwork
include("GVFNetwork.jl")
include("RNN.jl")
include("ForecastNetwork.jl")

export RTD, RTD_jacobian, TDLambda, TD, update!
include("Loss.jl")
include("Update.jl")
include("TimeseriesUpdates.jl")

export GradientGVFN
include("RGTD.jl")

export OnlineJointTD, OnlineTD_RNN, train_step!
include("RNN_updates.jl")


include("ActingPolicy.jl")

include("Environments.jl")

export jacobian, glorot_uniform, glorot_normal, StopGradient, get_clip_coeff
export FluxUtils, CycleWorldUtils, RingWorldUtils, CompassWorldUtils, TimeSeriesUtils
include("util.jl")

include("Agent.jl")

end
