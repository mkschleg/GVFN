import Flux

import Random
using DataStructures: CircularBuffer, isfull
using MinimalRLCore


mutable struct ForecastAgent{LU, C, O, F, H, Φ, Π, G} <: MinimalRLCore.AbstractAgent
    lu::LU
    chain::C
    opt::O
    τ::Int

    horde::G
    k::Array{Int, 1}
    k_idx::Array{Int, 1}
    
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}

    hidden_state_init::H

    π::Π
    
    s_t::Φ
    action::Int
    action_prob::Float32
end


function ForecastAgent(horde,
                       forecast_obj::Array{Int, 1},
                       forecast_obj_idx::Array{Int, 1},
                       chain,
                       opt,
                       τ,
                       feature_creator,
                       feature_size,
                       acting_policy::Π;
                       rng=Random.GLOBAL_RNG) where {Π<:AbstractActingPolicy}

    ####
    # Define Model
    ####
    k_perm = sortperm(forecast_obj)

    state_list, init_state = begin
        if needs_action_input(chain)
            (DataStructures.CircularBuffer{Tuple{Int, Array{Float32, 1}}}(τ+forecast_obj[k_perm][end]), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(τ+forecast_obj[k_perm][end]), zeros(Float32, 1))
        end
    end

    hidden_state_init = get_initial_hidden_state(chain)

    ForecastAgent(TD(),
                  chain,
                  opt,
                  τ,
                  horde,
                  forecast_obj,
                  forecast_obj_idx,
                  feature_creator,
                  state_list,
                  hidden_state_init,
                  acting_policy,
                  init_state,
                  0,
                  0.0f0)

end

function build_new_feat(agent::A, state, action) where {A<:ForecastAgent}
    if agent.s_t isa Tuple
        (action, agent.build_features(state, nothing))
    elseif agent.s_t isa AbstractArray
        agent.build_features(state, action)
    end
end


# build_new_feat(prev_state::Φ, state, action) where {Φ<:AbstractArray} = 
#     agent.build_features(state, action)

# build_new_feat(prev_state::Φ, state, action) where {Φ<:Tuple}= 
#     (action, agent.build_features(state, nothing))

function MinimalRLCore.start!(agent::ForecastAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

    # Fill the buffer w/ τ beginning copies of the starting state similar to GVFNs and RNNs.
    for i ∈ 1:agent.τ
        push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    end
    agent.hidden_state_init = get_initial_hidden_state(agent.chain)
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    return agent.action
    
end

function MinimalRLCore.step!(agent::ForecastAgent,
                             env_s_tp1,
                             r,
                             terminal,
                             rng=Random.GLOBAL_RNG)
    
    # Decide Action
    new_action, new_prob = agent.π(env_s_tp1, rng)
    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    
    if isfull(agent.state_list)
        targets = if agent.s_t isa Tuple
            getindex.(getindex.(agent.state_list, 2)[agent.τ .+ agent.k], agent.k_idx)
        else
            getindex.(agent.state_list[agent.τ .+ agent.k], agent.k_idx)
        end
        update!(agent.chain,
                agent.horde,
                agent.opt,
                agent.lu,
                agent.hidden_state_init,
                agent.state_list,
                env_s_tp1,
                agent.action,
                agent.action_prob;
                targets=targets,
                τ=agent.τ)
    end

    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.state_list)[end]

    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.state_list[1])
    
    agent.action = copy(new_action)
    agent.action_prob = new_prob
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)

    return (out_preds=Flux.data(out_preds)::Array{Float32, 1}, action=agent.action)
end


