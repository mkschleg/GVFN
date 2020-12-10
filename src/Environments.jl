


env_settings!(as::Reproduce.ArgParseSettings, env_type::Type{<:MinimalRLCore.AbstractEnvironment}) =
    throw("Set settings function for $(typeof(env_type))")


export CompassWorld, get_num_features
include("env/CompassWorld.jl")

export CycleWorld
include("env/CycleWorld.jl")

export MackeyGlass, MSO, SineWave, ACEA, Critterbot, CritterbotTPC, get_num_targets
include("env/TimeSeries.jl")

export RingWorld
include("env/RingWorld.jl")

export ContFourRooms
include("env/ContFourRooms.jl")

# export TrialABCEnv, ContABCEnv
# include("env/ABCEnv.jl")
