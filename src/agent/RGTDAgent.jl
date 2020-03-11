import Flux

import Random
import DataStructures

import MinimalRLCore

mutable struct RGTDAgent{LU<:AbstractGradUpdate, O, GVFN, F, H, Φ, Π, M, G} <: MinimalRLCore.AbstractAgent
    lu::LU
    opt::O
    gvfn::GVFN
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    π::Π
    action::Int
    action_prob::Float32
    model::M
    out_horde::Horde{G}
    preds_tp1::Array{Float32, 1}
end


function RGTDAgent(out_horde,
                   gvfn,
                   out_model,
                   lu,
                   opt,
                   τ,
                   feature_creator,
                   feature_size,
                   acting_policy)

    num_gvfs = length(gvfn.horde)
    s_t = if gvfn isa GradientGVFN_act
        (0, zeros(Float32, 1))
    else
        zeros(Float32, 1)
    end
    state_list = DataStructures.CircularBuffer{typeof(s_t)}(τ+1)
    
    hidden_state_init = zeros(Float32, num_gvfs)

    RGTDAgent(lu,
              opt,
              gvfn,
              feature_creator,
              state_list,
              hidden_state_init,
              # zeros(Float32, 1),
              s_t,
              acting_policy,
              -1, 0.0f0, out_model,
              out_horde, zeros(Float32, num_gvfs))

end

function build_new_feat(agent::A, state, action) where {A<:RGTDAgent}
    if agent.s_t isa Tuple
        (action, agent.build_features(state, nothing))
    elseif agent.s_t isa AbstractArray
        agent.build_features(state, action)
    end
end

function MinimalRLCore.start!(agent::RGTDAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

    # fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    
    agent.hidden_state_init .= zero(agent.hidden_state_init)
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    return agent.action
end

function MinimalRLCore.step!(agent::RGTDAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)

    # Decide Action
    new_action, new_prob = agent.π(env_s_tp1, rng)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))


    if DataStructures.isfull(agent.state_list)
        update!(agent.gvfn,
                agent.opt,
                agent.lu,
                agent.hidden_state_init,
                agent.state_list,
                env_s_tp1,
                agent.action,
                agent.action_prob)
    end
    preds = GVFN.roll(agent.gvfn,
                      agent.state_list,
                      agent.hidden_state_init,
                      GVFN.Prediction)
    
    agent.preds_tp1 .= preds[end]
    update!(agent.model,
            agent.out_horde,
            agent.opt,
            agent.lu,
            Flux.data.(preds),
            env_s_tp1,
            agent.action,
            agent.action_prob)


    out_preds = agent.model(preds[end])

    agent.action = copy(new_action)
    agent.action_prob = new_prob
    
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    agent.hidden_state_init .= Flux.data(preds[1])

    return (out_preds=Flux.data.(out_preds), action=agent.action, preds=preds[end])
end


function Base.print(io::IO, agent::RGTDAgent)
    print(io, Base.string(agent.preds_tp1))
end
