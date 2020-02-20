


agent_settings!(as::Reproduce.ArgParseSettings, agent::Type{<:MinimalRLCore.AbstractAgent}) =
    throw("Set settings function for $(typeof(agent))")

# Specialized Agents
include("agent/timeseries.jl")

# (Mostly) General Agents.
include("agent/FluxAgent.jl")
include("agent/ForecastAgent.jl")
# include("agent/ForecastActionAgent.jl")




