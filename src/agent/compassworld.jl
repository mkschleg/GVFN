
import JuliaRL

export CompassWorldAgent, CompassWorldRNNAgent

# import CycleWorldUtils
import Flux

import Random
import DataStructures

function get_action(state, env_state, rng=Random.GLOBAL_RNG)

    if state == ""
        state = "Random"
    end

    cwc = GVFN.CompassWorldConst
    

    if state == "Random"
        r = rand(rng)
        if r > 0.9
            state = "Leap"
        end
    end

    if state == "Leap"
        if env_state[cwc.WHITE] == 0.0
            state = "Random"
        else
            return state, (cwc.FORWARD, 1.0)
        end
    end
    r = rand(rng)
    if r < 0.2
        return state, (cwc.RIGHT, 0.2)
    elseif r < 0.4
        return state, (cwc.LEFT, 0.2)
    else
        return state, (cwc.FORWARD, 0.6)
    end
end

function get_action(rng=Random.GLOBAL_RNG)
    
    cwc = GVFN.CompassWorldConst
    r = rand(rng)
    if r < 0.2
        return cwc.RIGHT, 0.2
    elseif r < 0.4
        return cwc.LEFT, 0.2
    else
        return cwc.FORWARD, 0.6
    end
end

mutable struct CompassWorldAgent{O, T, F, H, Φ, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    opt::O
    gvfn::Flux.Recur{T}
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    action::Int64
    action_prob::Float64
    action_state::String
    model::M
    out_horde::Horde{G}
end


function CompassWorldAgent(parsed; rng=Random.GLOBAL_RNG)

    horde = CompassWorldUtils.get_horde(parsed)
    num_gvfs = length(horde)

    alg_string = parsed["alg"]
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func(Float64.(parsed["params"])...)
    τ=parsed["truncation"]

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["optparams"])...)

    act = FluxUtils.get_activation(parsed["act"])

    gvfn = GVFNetwork(num_gvfs, 19, horde; init=(dims...)->glorot_uniform(rng, dims...), σ_int=act)
    model = Linear(num_gvfs, 5; init=(dims...)->glorot_uniform(rng, dims...))
    out_horde = CompassWorldUtils.forward()

    state_list = DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1)
    # fill!(state_list, zeros(Float32, num_observations))
    hidden_state_init = zeros(Float32, num_gvfs)

    CompassWorldAgent(lu, opt, gvfn, CompassWorldUtils.build_features, state_list, hidden_state_init, zeros(Float32, 1), -1, 0.0, "", model, out_horde)

end

function JuliaRL.start!(agent::CompassWorldAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    agent.action_state, (agent.action, agent.action_prob) = get_action(agent.action_state, env_s_tp1, rng)

    fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    agent.hidden_state_init .= zero(agent.hidden_state_init)
    agent.s_t = copy(env_s_tp1)
    return agent.action
end

function JuliaRL.step!(agent::CompassWorldAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))

    update!(agent.gvfn, agent.opt, agent.lu, agent.hidden_state_init, agent.state_list, env_s_tp1, agent.action, agent.action_prob)

    reset!(agent.gvfn, agent.hidden_state_init)
    preds = agent.gvfn.(agent.state_list)

    update!(agent.model, agent.out_horde, agent.opt, agent.lu, Flux.data.(preds), env_s_tp1, agent.action, agent.action_prob)

    out_preds = agent.model(preds[end])

    agent.s_t .= env_s_tp1
    agent.hidden_state_init .= Flux.data(preds[1])

    agent.action_state, (agent.action, agent.action_prob) = get_action(agent.action_state, agent.s_t, rng)

    return Flux.data.(out_preds), agent.action[1]
end

JuliaRL.get_action(agent::CompassWorldAgent, state) = agent.action


mutable struct CompassWorldRNNAgent{O, T, F, H, Φ, M, G} <: JuliaRL.AbstractAgent
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
    action_state::String
    horde::Horde{G}
end

function CompassWorldRNNAgent(parsed; rng=Random.GLOBAL_RNG)

    horde = CompassWorldUtils.get_horde(parsed)
    num_gvfs = length(horde)

    fc = CompassWorldUtils.ActionTileFeatureCreator()

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)
    rnn = FluxUtils.construct_rnn(CompassWorldUtils.feature_size(fc), parsed; init=(dims...)->glorot_uniform(rng, dims...))
    out_model = Flux.Dense(parsed["numhidden"], length(horde); initW=(dims...)->glorot_uniform(rng, dims...))

    state_list =  DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1)
    hidden_state_init = GVFN.get_initial_hidden_state(rnn)

    CompassWorldRNNAgent(TD(), opt, rnn, out_model, fc, state_list, hidden_state_init, zeros(Float32, 1), 1, 0.0, "", horde)

end

function JuliaRL.start!(agent::CompassWorldRNNAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    agent.action_state, (agent.action, agent.action_prob) = get_action(agent.action_state, env_s_tp1, rng)

    fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    agent.hidden_state_init = FluxUtils.get_initial_hidden_state(agent.rnn)
    agent.s_t = copy(env_s_tp1)
    return agent.action
end


function JuliaRL.step!(agent::CompassWorldRNNAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

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
    agent.action_state, (agent.action, agent.action_prob) = get_action(agent.action_state, agent.s_t, rng)

    return Flux.data.(out_preds), agent.action
end

JuliaRL.get_action(agent::CompassWorldRNNAgent, state) = agent.action

