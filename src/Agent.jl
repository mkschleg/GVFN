


agent_settings!(as::Reproduce.ArgParseSettings, agent::Type{<:JuliaRL.AbstractAgent}) =
    throw("Set settings function for $(typeof(agent))")


# include("agent/cycleworld.jl")
# include("agent/compassworld.jl")
include("agent/mackeyglass.jl")


include("agent/GVFNAgent.jl")
include("agent/GVFNActionAgent.jl")
include("agent/RNNAgent.jl")
include("agent/RNNActionAgent.jl")

include("agent/ForecastAgent.jl")
include("agent/ForecastActionAgent.jl")




