
__precompile__(true)

module GVFN
import Flux
using Reexport

import Reproduce

@reexport using MinimalRLCore

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
    cumulant,
    discount,
    policy,
    Horde,
    NullPolicy,
    PersistentPolicy,
    PredictionConditionalPolicy,
    ConstantDiscount,
    StateTerminationDiscount,
    FeatureCumulant,
    PredictionCumulant,
    ScaledCumulant,
    NormalizedCumulant

include("GVF.jl")

export GVFNetwork, GVFActionNetwork, reset!, get, RNNActionLayer, ForecastNetwork, GVFR, GVFRAction
include("GVFNetwork.jl")
include("RNN.jl")
include("ForecastNetwork.jl")

export RTD, RTD_jacobian, TDLambda, TD, update!
include("Loss.jl")
include("Update.jl")
include("FluxUpdate.jl")
include("TimeseriesUpdates.jl")

export GradientGVFN
include("RGTD.jl")
include("RGTD_act.jl")

include("ActingPolicy.jl")

include("Environments.jl")

export jacobian, glorot_uniform, glorot_normal, StopGradient, get_clip_coeff
export FluxUtils, CycleWorldUtils, RingWorldUtils, CompassWorldUtils, TimeSeriesUtils
include("util.jl")

include("Agent.jl")

end
