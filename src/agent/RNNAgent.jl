import Flux

import Random
import DataStructures

mutable struct RNNAgent{O, T, F, H, Φ, Π, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    opt::O
    rnn::Flux.Recur{T}
    out_model::M
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    π::Π
    action::Int64
    action_prob::Float64
    horde::Horde{G}
    prev_action_or_not::Bool
end




function RNNAgent(out_horde,
                  feature_creator,
                  feature_size,
                  acting_policy::Π,
                  parsed;
                  rng=Random.GLOBAL_RNG,
                  init_func=(dims...)->glorot_uniform(rng, dims...)) where {Π<:AbstractActingPolicy}

    num_gvfs = length(out_horde)

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)
    rnn = FluxUtils.construct_rnn(feature_size, parsed; init=init_func)
    out_model = Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func)

    prev_action_or_not = get(parsed, "prev_action_or_not", false)

    state_list =  DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1)
    hidden_state_init = GVFN.get_initial_hidden_state(rnn)

    RNNAgent(TD(), opt,
             rnn, out_model,
             feature_creator,
             state_list,
             hidden_state_init,
             zeros(Float32, 1),
             acting_policy,
             1, 0.0, out_horde,
             prev_action_or_not)

end

function JuliaRL.start!(agent::RNNAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    # agent.action = sample(rng, agent.π, env_s_tp1)
    # agent.action_prob = get(agent.π, env_s_tp1, agent.action)
    agent.action, agent.action_prob = agent.π(env_s_tp1, rng)
    
    fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    agent.hidden_state_init = FluxUtils.get_initial_hidden_state(agent.rnn)
    agent.s_t = copy(env_s_tp1)
    return agent.action
end


function JuliaRL.step!(agent::RNNAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)


    # new_action = sample(rng, agent.π, env_s_tp1)
    new_action, new_prob = agent.π(env_s_tp1, rng)

    if agent.prev_action_or_not
        push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    else
        push!(agent.state_list, agent.build_features(env_s_tp1, new_action))
    end

    # RNN update function
    update!(agent.out_model, agent.rnn,
            agent.horde, agent.opt,
            agent.lu, agent.hidden_state_init,
            agent.state_list, env_s_tp1,
            agent.action, agent.action_prob)
    # End update function

    Flux.truncate!(agent.rnn)
    reset!(agent.rnn, agent.hidden_state_init)
    rnn_out = agent.rnn.(agent.state_list)
    out_preds = agent.out_model(rnn_out[end])

    agent.hidden_state_init =
        FluxUtils.get_next_hidden_state(agent.rnn, agent.hidden_state_init, agent.state_list[1])
    agent.s_t .= env_s_tp1

    
    agent.action = copy(new_action)
    agent.action_prob = new_prob

    return Flux.data.(out_preds), agent.action
end

JuliaRL.get_action(agent::RNNAgent, state) = agent.action
