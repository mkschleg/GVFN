
import JuliaRL

export RingWorldAgent, RingWorldRNNAgent

# import CycleWorldUtils
import Flux

import Random
import DataStructures

function get_action(env_state, rng=Random.GLOBAL_RNG)
    rn = rand(rng)
    if rn < 0.5
        return 1, 0.5
    else
        return 2, 0.5
    end
end


mutable struct RingWorldAgent{O, T, F, H, Φ, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    opt::O
    gvfn::Flux.Recur{T}
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    action::Int64
    action_prob::Float64
    model::M
    out_horde::Horde{G}
end


function RingWorldAgent(parsed; rng=Random.GLOBAL_RNG)

    horde = RingWorldUtils.get_horde(parsed)
    num_gvfs = length(horde)

    alg_string = parsed["alg"]
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func(Float64.(parsed["params"])...)
    τ=parsed["truncation"]

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["optparams"])...)

    act = FluxUtils.get_activation(parsed["act"])

    fc = RingWorldUtils.StandardFeatureCreator()
    if parsed["feature"] == "action"
        fc = RingWorldUtils.ActionTileFeatureCreator()
    end

    gvfn = GVFNetwork(num_gvfs, RingWorldUtils.feature_size(fc), horde; init=(dims...)->glorot_uniform(rng, dims...), σ_int=act)
    model = Linear(num_gvfs, 5; init=(dims...)->glorot_uniform(rng, dims...))
    out_horde = RingWorldUtils.forward()

    state_list = DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1)
    # fill!(state_list, zeros(Float32, num_observations))
    hidden_state_init = zeros(Float32, num_gvfs)

    RingWorldAgent(lu, opt, gvfn, fc, state_list, hidden_state_init, zeros(Float32, 1), -1, 0.0, model, out_horde)

end

function JuliaRL.start!(agent::RingWorldAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    agent.action, agent.action_prob = get_action(env_s_tp1, rng)

    fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    agent.hidden_state_init .= zero(agent.hidden_state_init)
    agent.s_t = copy(env_s_tp1)
    return agent.action
end

function JuliaRL.step!(agent::RingWorldAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))

    update!(agent.gvfn, agent.opt, agent.lu, agent.hidden_state_init, agent.state_list, env_s_tp1, agent.action, agent.action_prob)

    reset!(agent.gvfn, agent.hidden_state_init)
    preds = agent.gvfn.(agent.state_list)

    update!(agent.model, agent.out_horde, agent.opt, agent.lu, Flux.data.(preds), env_s_tp1, agent.action, agent.action_prob)

    out_preds = agent.model(preds[end])

    agent.s_t .= env_s_tp1
    agent.hidden_state_init .= Flux.data(preds[1])

    agent.action, agent.action_prob = get_action(agent.s_t, rng)

    return Flux.data.(out_preds), agent.action
end

JuliaRL.get_action(agent::RingWorldAgent, state) = agent.action


mutable struct RingWorldRNNAgent{O, T, F, H, Φ, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    opt::O
    rnn::Flux.Recur{T}
    out_model::M
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    action::Int64
    action_prob::Float64
    horde::Horde{G}
end

function RingWorldRNNAgent(parsed; rng=Random.GLOBAL_RNG)

    horde = RingWorldUtils.get_horde(parsed)
    num_gvfs = length(horde)

    fc = RingWorldUtils.StandardFeatureCreator()
    if parsed["feature"] == "action"
        fc = RingWorldUtils.ActionTileFeatureCreator()
    end

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)
    rnn = FluxUtils.construct_rnn(RingWorldUtils.feature_size(fc), parsed; init=(dims...)->glorot_uniform(rng, dims...))
    out_model = Flux.Dense(parsed["numhidden"], length(horde); initW=(dims...)->glorot_uniform(rng, dims...))

    state_list =  DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1)
    hidden_state_init = GVFN.get_initial_hidden_state(rnn)

    RingWorldRNNAgent(TD(), opt, rnn, out_model, fc, state_list, hidden_state_init, zeros(Float32, 1), 1, 0.0, horde)

end

function JuliaRL.start!(agent::RingWorldRNNAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    agent.action, agent.action_prob = get_action(env_s_tp1, rng)

    fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    agent.hidden_state_init = FluxUtils.get_initial_hidden_state(agent.rnn)
    agent.s_t = copy(env_s_tp1)
    return agent.action
end


function JuliaRL.step!(agent::RingWorldRNNAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))

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

    agent.hidden_state_init = FluxUtils.get_next_hidden_state(agent.rnn, agent.hidden_state_init, agent.state_list[1])
    agent.s_t .= env_s_tp1
    agent.action, agent.action_prob = get_action(agent.s_t, rng)

    return Flux.data.(out_preds), agent.action
end

JuliaRL.get_action(agent::RingWorldRNNAgent, state) = agent.action

