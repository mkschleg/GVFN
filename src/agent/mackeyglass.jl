export MackeyGlassAgent, MackeyGlassRNNAgent, predict!

import Flux
import Random
import DataStructures

#import JuliaRL

mutable struct MackeyGlassAgent{GVFNOpt,ModelOpt, T, H, Φ, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    gvfn_opt::GVFNOpt
    model_opt::ModelOpt
    gvfn::Flux.Recur{T}
    state_list::DataStructures.CircularBuffer{Φ}
    pbuff::Vector{Φ}
    targetbuff::Vector{Float64}
    hidden_state_init::H
    s_t::Φ
    model::M
    out_horde::Horde{G}

    horizon::Int
    step::Int
    batchsize::Int
    ϕbuff::DataStructures.CircularBuffer{Vector{Float64}}
end


function MackeyGlassAgent(parsed; rng=Random.GLOBAL_RNG)

    horde = TimeSeriesUtils.get_horde(parsed)
    num_gvfs = length(horde)

    alg_string = parsed["alg"]
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func()

    gvfn_opt_string = parsed["gvfn_opt"]
    gvfn_opt_func = getproperty(Flux, Symbol(gvfn_opt_string))
    gvfn_opt = gvfn_opt_func(parsed["gvfn_stepsize"])
    batchsize=parsed["batchsize"]

    model_opt_string = parsed["model_opt"]
    model_opt_func = getproperty(Flux, Symbol(model_opt_string))
    model_opt = model_opt_func(parsed["model_stepsize"])

    act = FluxUtils.get_activation(parsed["act"])
    init_func = (dims...)->glorot_uniform(rng, dims...)
    gvfn = GVFNetwork(num_gvfs, 1, horde; init=init_func, σ_int=act)
    model = Flux.Chain(
        Flux.Dense(num_gvfs,num_gvfs,relu; initW=init_func),
        Flux.Dense(num_gvfs, 1; initW=init_func);
    )
    out_horde = Horde([GVF(FeatureCumulant(1),ConstantDiscount(0.0), NullPolicy())])

    state_list = DataStructures.CircularBuffer{Array{Float64, 1}}(batchsize+1)
    hidden_state_init = zeros(Float64, num_gvfs)

    targetbuff = Float64[]
    pbuff = Array{Float64,1}[]

    horizon = Int(parsed["horizon"])
    ϕbuff = DataStructures.CircularBuffer{Vector{Float64}}(horizon)

    return MackeyGlassAgent(lu, gvfn_opt, model_opt, gvfn, state_list, pbuff, targetbuff, hidden_state_init, zeros(Float64, 1), model, out_horde, horizon, 0, batchsize,ϕbuff)
end

function start!(agent::MackeyGlassAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    fill!(agent.state_list, zeros(1))
    push!(agent.state_list, env_s_tp1)
    agent.hidden_state_init .= zero(agent.hidden_state_init)
    agent.s_t = copy(env_s_tp1)

    agent.step+=1
end

function step!(agent::MackeyGlassAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    push!(agent.state_list, env_s_tp1)
    if agent.step % agent.batchsize == 0
        update!(agent.gvfn, agent.gvfn_opt, agent.lu, agent.hidden_state_init, agent.state_list, env_s_tp1)
    end

    reset!(agent.gvfn, agent.hidden_state_init)
    preds = agent.gvfn.(agent.state_list)

    push!(agent.ϕbuff, preds[end-1].data)
    if agent.step>=agent.horizon
        push!(agent.pbuff, popfirst!(agent.ϕbuff))
        push!(agent.targetbuff, env_s_tp1[1])
        if length(agent.targetbuff) == agent.batchsize
            update!(agent.model, agent.out_horde, agent.model_opt, agent.lu, agent.pbuff, agent.targetbuff)

            agent.targetbuff = Float64[]
            agent.pbuff = Vector{Float64}[]
        end
    end

    out_preds = agent.model(preds[end].data)

    agent.s_t .= env_s_tp1
    agent.hidden_state_init .= preds[1].data

    agent.step+=1

    return out_preds.data
end

function predict!(agent::MackeyGlassAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG,kwargs...)
    # for validation/test; predict, updating hidden states, but don't update models
    push!(agent.state_list, env_s_tp1)

    reset!(agent.gvfn, agent.hidden_state_init)
    preds = agent.gvfn.(agent.state_list)

    out_preds = agent.model(preds[end])

    agent.s_t .= env_s_tp1
    agent.hidden_state_init .= preds[1].data

    return out_preds.data

end

mutable struct MackeyGlassRNNAgent{O, T, F, H, Φ, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    opt::O
    rnn::Flux.Recur{T}
    out_model::M
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    action::Int64
    action_prob::Float64
    action_state::String
    horde::Horde{G}
end

function MackeyGlassRNNAgent(parsed; rng=Random.GLOBAL_RNG)

    horde = TimeSeriesUtils.get_horde(parsed)
    num_gvfs = length(horde)

    τ=parsed["rnn_tau"]

    opt = getproperty(Flux, parsed["rnn_opt"])(parsed["rnn_lr"])

    rnn = FluxUtils.construct_rnn(1, parsed; init=(dims...)->glorot_uniform(rng, dims...))
    out_model = Flux.Dense(parsed["rnn_nhidden"], length(horde); initW=(dims...)->glorot_uniform(rng, dims...))

    state_list =  DataStructures.CircularBuffer{Array{Float64, 1}}(τ+1)
    hidden_state_init = GVFN.get_initial_hidden_state(rnn)

    MackeyGlassRNNAgent(BatchTD(), opt, rnn, out_model, state_list, hidden_state_init, zeros(Float64, 1), 1, 0.0, "", horde)

end

function JuliaRL.start!(agent::MackeyGlassRNNAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    agent.action_state, (agent.action, agent.action_prob) = get_action(agent.action_state, agent.s_t, rng)

    fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    agent.hidden_state_init = FluxUtils.get_initial_hidden_state(agent.rnn)
    agent.s_t = copy(env_s_tp1)
    return agent.action
end


function step!(agent::MackeyGlassRNNAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

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

JuliaRL.get_action(agent::MackeyGlassRNNAgent, state) = agent.action

