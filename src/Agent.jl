


include("agent/cycleworld.jl")
include("agent/compassworld.jl")
include("agent/mackeyglass.jl")


# mutable struct GVFNAgent{O, T, F, H, Φ, M, G, P} <: JuliaRL.AbstractAgent
#     lu::LearningUpdate
#     opt::O
#     gvfn::Flux.Recur{T}
#     build_features::F
#     state_list::DataStructures.CircularBuffer{Φ}
#     hidden_state_init::H
#     s_t::Φ
#     action::Int64
#     model::M
#     out_horde::Horde{G}
#     μ::P
# end


# function JuliaRL.start!(agent::GVFNAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)
#     agent.s_t .= env_s_tp1
#     fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1))))
#     push!(agent.state_list, agent.build_features(env_s_tp1))
#     agent.hidden_state_init .= zero(agent.hidden_state_init)
#     agent.s_t .= env_s_tp1
#     return agent.action
# end

# function JuliaRL.step!(agent::GVFNAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

#     push!(agent.state_list, agent.build_features(env_s_tp1))

#     update!(agent.gvfn, agent.opt, agent.lu, agent.hidden_state_init, agent.state_list, env_s_tp1)

#     reset!(agent.gvfn, agent.hidden_state_init)
#     preds = agent.gvfn.(agent.state_list)

#     update!(agent.model, agent.out_horde, agent.opt, agent.lu, Flux.data.(preds), env_s_tp1)

#     out_preds = agent.model(preds[end])

#     agent.s_t .= env_s_tp1
#     agent.hidden_state_init .= Flux.data(preds[1])
#     agent.action = agent.μ(state)

#     return Flux.data.(out_preds), agent.action
# end

# JuliaRL.get_action(agent::GVFNAgent, state) = agent.μ(state)

