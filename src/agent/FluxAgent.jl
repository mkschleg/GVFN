
using MinimalRLCore
import Flux
import Random
import DataStructures

mutable struct FluxAgent{LU<:LearningUpdate, O, C, F, H, Φ, Π, G} <: MinimalRLCore.AbstractAgent
    lu::LU
    opt::O
    chain::C
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    π::Π
    action::Int
    action_prob::Float32
    horde::Horde{G}
end

function FluxAgent(out_horde,
                   chain,
                   opt,
                   τ,
                   feature_creator,
                   feature_size,
                   acting_policy::Π;
                   rng=Random.GLOBAL_RNG) where {Π<:AbstractActingPolicy}

    num_gvfs = length(out_horde)

    state_list, init_state = begin
        if needs_action_input(chain)
            (DataStructures.CircularBuffer{Tuple{Int, Array{Float32, 1}}}(τ+1), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1), zeros(Float32, 1))
        end
    end
    
    hidden_state_init = get_initial_hidden_state(chain)
    
    FluxAgent(TD(),
              opt,
              chain,
              feature_creator,
              state_list,
              hidden_state_init,
              init_state,
              acting_policy,
              1, 0.0f0, out_horde)

end

function agent_settings!(as::Reproduce.ArgParseSettings,
                         env_type::Type{FluxAgent})
    FluxUtils.opt_settings!(as)
    FluxUtils.rnn_settings!(as)
end


build_new_feat(agent::FluxAgent{LU, O, C, F, H, Φ, Π, G}, state, action) where {LU<:LearningUpdate, O, C, F, H, Φ, Π, G} = 
    agent.build_features(state, action)

build_new_feat(agent::FluxAgent{LU, O, C, F, H, Φ, Π, G}, state, action) where {LU<:LearningUpdate, O, C, F, H, Φ<:Tuple, Π, G}= 
    (action, agent.build_features(state, nothing))

function MinimalRLCore.start!(agent::FluxAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

    fill!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.chain)
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    return agent.action
end


function MinimalRLCore.step!(agent::FluxAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)


    # new_action = sample(rng, agent.π, env_s_tp1)
    new_action, new_prob = agent.π(env_s_tp1, rng)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    
    # RNN update function
    update!(agent.chain,
            agent.horde,
            agent.opt,
            agent.lu,
            agent.hidden_state_init,
            agent.state_list,
            env_s_tp1,
            agent.action,
            agent.action_prob)
    # End update function

    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.state_list)[end]

    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.state_list[1])

    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    agent.action = copy(new_action)
    agent.action_prob = new_prob
    

    return (out_preds=Flux.data(out_preds)::Array{Float32, 1}, action=agent.action)
end

# MinimalRLCore.get_action(agent::FluxAgent, state) = agent.action
