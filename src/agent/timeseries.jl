export TimeSeriesAgent, TimeSeriesRNNAgent, predict!

import Flux
import Random
import DataStructures
import MinimalRLCore

mutable struct TimeSeriesFluxAgent{O1, O2, C, F, H, Φ} <: MinimalRLCore.AbstractAgent
    lu::LearningUpdate
    model_opt::O1
    gvfn_opt::O2
    model::C
    build_features::F

    
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    
    horizon::Int
    step::Int
    batchsize::Int
end

function TimeSeriesFluxAgent(parsed; rng=Random.GLOBAL_RNG)

    
    horde = TimeSeriesUtils.get_horde(parsed)
    num_gvfs = length(horde)

    alg_string = parsed["alg"]
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func()

    gvfn_opt_string = parsed["gvfn_opt"]
    gvfn_opt_func = getproperty(Flux, Symbol(gvfn_opt_string))
    gvfn_opt = gvfn_opt_func(parsed["gvfn_stepsize"])
    


    model_opt_string = parsed["model_opt"]
    model_opt_func = getproperty(Flux, Symbol(model_opt_string))
    model_opt = model_opt_func(parsed["model_stepsize"])

    normalizer = TimeSeriesUtils.getNormalizer(parsed)

    init_func = (dims...)->glorot_uniform(rng, dims...)

    model = Flux.Chain(
        GVFR_RNN(1, horde, identity; init=init_func),
        Flux.data,
        Flux.Dense(num_gvfs, num_gvfs, relu; initW=init_func),
        Flux.Dense(num_gvfs, 1; initW=init_func)
    )

    horizon = Int(parsed["horizon"])
    batchsize=parsed["batchsize"]
    
    state_list, init_state = 
        (DataStructures.CircularBuffer{Array{Float32, 1}}(batchsize + horizon + 1), zeros(Float32, 1))

    hidden_state_init = get_initial_hidden_state(model)


    TimeSeriesFluxAgent(
        lu,
        model_opt,
        gvfn_opt,
        model,
        normalizer,
        state_list,
        hidden_state_init,
        init_state,
        horizon,
        0,
        batchsize
    )
end

function MinimalRLCore.start!(agent::TimeSeriesFluxAgent, env_s_tp1, rng=Random.GLOBAL_RNG)


    agent.s_t .= agent.build_features(env_s_tp1)
    
    fill!(agent.state_list, agent.build_features(env_s_tp1))
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    
    agent.step+=1
end

function MinimalRLCore.step!(agent::TimeSeriesFluxAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)

    horizon = agent.horizon
    batchsize = agent.batchsize
    model = agent.model

    push!(agent.state_list, agent.build_features(env_s_tp1))
    

    push!(agent.hidden_states, get_hidden_state(agent.model))

    if agent.step>=agent.horizon
        push!(agent.batch_h, popfirst!(agent.hidden_states))
        push!(agent.batch_obs, env_s_tp1[1])
        if length(agent.batch_obs) == agent.batchsize
            update!(agent.model, agent.out_horde, agent.model_opt, agent.lu, agent.batch_h, agent.batch_obs)

            agent.batch_obs = Float64[]
            agent.batch_h = Vector{Float64}[]
        end
    end

    # don't judge me
    stp1 = agent.normalizer(env_s_tp1)
    v_tp1 = agent.gvfn(stp1,agent.h).data
    c, Γ, _ = get(agent.horde, nothing, env_s_tp1, v_tp1)
    push!(agent.batch_target, c .+ Γ.*v_tp1)
    push!(agent.batch_phi, copy(agent.s_t))
    push!(agent.batch_hidden, copy(agent.h))
    if length(agent.batch_phi) == agent.batchsize
        update!(agent.gvfn, agent.gvfn_opt, agent.lu, agent.batch_hidden, agent.batch_phi, agent.batch_target)

        agent.batch_phi = Vector{Float64}[]
        agent.batch_hidden = Vector{Float64}[]
        agent.batch_target = Vector{Float64}[]
    end

    agent.s_t .= stp1
    agent.h .= v_tp1
    agent.step+=1

    return agent.model(v_tp1).data
end

function predict!(agent::TimeSeriesFluxAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG,kwargs...)
    # for validation/test; predict, updating hidden states, but don't update models

    push!(agent.state_list, agent.build_features(env_s_tp1))
    return agent.model.(agent.state_list)[end].data

end

# ==================
# --- GVFN AGENT ---
# ==================

mutable struct TimeSeriesAgent{GVFNOpt,ModelOpt, J, H, Φ, M, G1, G2, N} <: MinimalRLCore.AbstractAgent
    lu::LearningUpdate
    gvfn_opt::GVFNOpt
    model_opt::ModelOpt
    gvfn::J
    normalizer::N

    batch_phi::Vector{Φ}
    batch_target::Vector{Φ}
    batch_hidden::Vector{Φ}
    batch_h::Vector{Φ}
    batch_obs::Vector{Float64}

    hidden_states::Vector{Φ}

    h::H
    s_t::Φ
    model::M
    horde::Horde{G1}
    out_horde::Horde{G2}

    horizon::Int
    step::Int
    batchsize::Int
end


function TimeSeriesAgent(parsed; rng=Random.GLOBAL_RNG)

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

    normalizer = TimeSeriesUtils.getNormalizer(parsed)

    init_func = (dims...)->glorot_uniform(rng, dims...)
    gvfn = JankyGVFLayer(1, num_gvfs; init=init_func)
    model = Flux.Chain(
        Flux.Dense(num_gvfs,num_gvfs,relu; initW=init_func),
        Flux.Dense(num_gvfs, 1; initW=init_func);
    )
    out_horde = Horde([GVF(FeatureCumulant(1),ConstantDiscount(0.0), NullPolicy())])

    # gvfn buffers
    batch_phi = Vector{Float64}[]
    batch_target = Vector{Float64}[]
    batch_hidden = Vector{Float64}[]
    hidden_states = Vector{Float64}[]

    hidden_state_init = zeros(Float64, num_gvfs)

    batch_obs = Float64[]
    batch_h = Array{Float64,1}[]

    horizon = Int(parsed["horizon"])

    return TimeSeriesAgent(lu, gvfn_opt, model_opt, gvfn, normalizer, batch_phi, batch_target, batch_hidden, batch_h, batch_obs, hidden_states, hidden_state_init, zeros(Float64, 1), model, horde, out_horde, horizon, 0, batchsize)
end

function MinimalRLCore.start!(agent::TimeSeriesAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    stp1 = agent.normalizer(env_s_tp1)

    agent.h .= zero(agent.h)
    agent.h .= agent.gvfn(stp1, agent.h).data
    agent.s_t .= stp1

    agent.step+=1
end

function MinimalRLCore.step!(agent::TimeSeriesAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)
    push!(agent.hidden_states, copy(agent.h))

    if agent.step>=agent.horizon
        push!(agent.batch_h, popfirst!(agent.hidden_states))
        push!(agent.batch_obs, env_s_tp1[1])
        if length(agent.batch_obs) == agent.batchsize
            update!(agent.model, agent.out_horde, agent.model_opt, agent.lu, agent.batch_h, agent.batch_obs)

            agent.batch_obs = Float64[]
            agent.batch_h = Vector{Float64}[]
        end
    end

    # don't judge me
    stp1 = agent.normalizer(env_s_tp1)
    v_tp1 = agent.gvfn(stp1,agent.h).data
    c, Γ, _ = get(agent.horde, nothing, env_s_tp1, v_tp1)
    push!(agent.batch_target, c .+ Γ.*v_tp1)
    push!(agent.batch_phi, copy(agent.s_t))
    push!(agent.batch_hidden, copy(agent.h))
    if length(agent.batch_phi) == agent.batchsize
        update!(agent.gvfn, agent.gvfn_opt, agent.lu, agent.batch_hidden, agent.batch_phi, agent.batch_target)

        agent.batch_phi = Vector{Float64}[]
        agent.batch_hidden = Vector{Float64}[]
        agent.batch_target = Vector{Float64}[]
    end

    agent.s_t .= stp1
    agent.h .= v_tp1
    agent.step+=1

    return agent.model(v_tp1).data
end

function predict!(agent::TimeSeriesAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG,kwargs...)
    # for validation/test; predict, updating hidden states, but don't update models

    agent.h .= agent.gvfn(env_s_tp1, agent.h).data
    return agent.model(agent.h).data

end

# =================
# --- RNN AGENT ---
# =================

mutable struct TimeSeriesRNNAgent{L,O, C, H, Φ} <: MinimalRLCore.AbstractAgent where {L<:LearningUpdate}
    lu::L
    opt::O
    chain::C

    obs_sequence::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H

    h_buff::DataStructures.CircularBuffer{H}
    obs_buff::DataStructures.CircularBuffer{Vector{Φ}}

    batch_h::Vector{H}
    batch_obs::Vector{Vector{Φ}}
    batch_target::Vector{Φ}

    horizon::Int
    batchsize::Int
end

# Convenient type aliases
Hidden_t = IdDict{Any,Any}
Obs_t = Vector{Float32}

function TimeSeriesRNNAgent(parsed; rng=Random.GLOBAL_RNG)

    # hyperparameters
    horizon=parsed["horizon"]
    batchsize = parsed["batchsize"]
    nhidden=parsed["rnn_nhidden"]
    τ=parsed["rnn_tau"]
    lr = parsed["rnn_lr"]

    # build model
    opt = getproperty(Flux, Symbol(parsed["rnn_opt"]))(lr)
    cell = getproperty(Flux, Symbol(parsed["rnn_cell"]))
    chain = Flux.Chain(
        cell(1, nhidden; init=(dims...)->glorot_uniform(rng, dims...)),
        Flux.Dense(parsed["rnn_nhidden"], 1 ; initW=(dims...)->glorot_uniform(rng, dims...))
    )

    obs_sequence = DataStructures.CircularBuffer{Obs_t}(τ+1)
    hidden_state_init = GVFN.get_initial_hidden_state(chain)

    # buffers for temporal offsets
    h_buff = DataStructures.CircularBuffer{Hidden_t}(horizon)
    obs_buff = DataStructures.CircularBuffer{Vector{Obs_t}}(horizon)

    # buffers for batches
    batch_obs, batch_h, batch_target = newBatch()

    TimeSeriesRNNAgent(BatchTD(),
                       opt,
                       chain,

                       obs_sequence,
                       hidden_state_init,

                       h_buff,
                       obs_buff,

                       batch_h,
                       batch_obs,
                       batch_target,

                       horizon,
                       batchsize)

end

function newBatch()
    # get empty batch buffers
    batch_obs = Vector{Obs_t}[]
    batch_h  = Hidden_t[]
    batch_target = Obs_t[]
    return batch_obs, batch_h, batch_target
end

function resetBatch!(agent::TimeSeriesRNNAgent)
    # reset the agent's batch buffers
    agent.batch_obs, agent.batch_h, agent.batch_target = newBatch()
end

function MinimalRLCore.start!(agent::TimeSeriesRNNAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    # init observation sequence
    fill!(agent.obs_sequence, copy(env_s_tp1))

    # init hidden state
    agent.hidden_state_init = get_initial_hidden_state(agent.chain)
end


function MinimalRLCore.step!(agent::TimeSeriesRNNAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)

    # Update state seq
    push!(agent.obs_sequence, copy(env_s_tp1))

    # copy state sequence/hidden state into temporal offset buffers
    push!(agent.obs_buff, copy(agent.obs_sequence))
    push!(agent.h_buff, copy(agent.hidden_state_init))

    # Update =====================================================
    if DataStructures.isfull(agent.obs_buff)

        # Add target, hidden state, and observation sequence to batch
        # ---| Target = most-recent observation; obs/hidden state = earliest in the buffer
        push!(agent.batch_target, copy(env_s_tp1))
        push!(agent.batch_obs, copy(agent.obs_buff[1]))
        push!(agent.batch_h, copy(agent.h_buff[1]))

        if length(agent.batch_target) == agent.batchsize
            update!(agent.chain,
                    agent.opt,
                    agent.lu,
                    agent.batchsize,
                    agent.batch_h,
                    agent.batch_obs,
                    agent.batch_target)


            # Reset the batch buffers
            resetBatch!(agent)
        end
    end

    # Predict ====================================================
    # Get RNN output/predictions
    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.obs_sequence)[end]

    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.obs_sequence[1])

    # Prediction for time t
    return out_preds.data
end

function predict!(agent::TimeSeriesRNNAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG,kwargs...)
    # for validation/test; predict, updating hidden states, but don't update models

    # Update the sequence of observations
    push!(agent.obs_sequence, env_s_tp1)

    # reset the chain's initial hidden state and run through the observation sequence
    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.obs_sequence)[end]

    # update the hidden state
    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.obs_sequence[1])

    return out_preds.data
end
