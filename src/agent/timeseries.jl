export TimeSeriesGVFNAgent, TimeSeriesRNNAgent, TimeSeriesAuxTaskAgent,
    TimeSeriesOriginalRNNAgent, TimeSeriesOriginalAuxTaskAgent,
    predict!

import Flux
import Random
import DataStructures
import MinimalRLCore

# =================
# --- FLUX AGENT ---
# =================

mutable struct TimeSeriesAgent{L, O, C, N, H, Φ} <: MinimalRLCore.AbstractAgent where {L<:LearningUpdate}
    lu::L
    opt::O
    chain::C
    normalizer::N

    obs_sequence::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H

    h_buff::DataStructures.CircularBuffer{H}
    obs_buff::DataStructures.CircularBuffer{Vector{Φ}}

    batch_h::Vector{H}
    batch_obs::Vector{Vector{Φ}}
    batch_gvfn_target::Vector{Φ}
    batch_model_target::Vector{Φ}

    horizon::Int
    batchsize::Int
    model_clip_coeff::Float32
end

# Convenient type aliases
Hidden_t = IdDict{Any,Any}
Obs_t = Vector{Float32}

function TimeSeriesGVFNAgent(parsed; rng=Random.GLOBAL_RNG)


    # ==========================================
    # hyperparameters
    # ==========================================
    alg_string = parsed["update_fn"]
    horizon = Int(parsed["horizon"])
    batchsize=parsed["batchsize"]

    τ=parsed["gvfn_tau"]
    gvfn_opt_string = parsed["gvfn_opt"]
    gvfn_stepsize = parsed["gvfn_stepsize"]

    model_clip_coeff = Float32(parsed["model_clip_coeff"])
    # ==========================================


    # =============================================================
    # Instantiations
    # =============================================================

    # GVFN learning update
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func()

    # Optimizers
    opt = begin
        gvfn_opt_func = getproperty(Flux, Symbol(gvfn_opt_string))
        gvfn_opt = gvfn_opt_func(gvfn_stepsize)

        if "model_opt" ∈ keys(parsed)
            model_opt_string = parsed["model_opt"]
            model_stepsize = parsed["model_stepsize"]

            model_opt_func = getproperty(Flux, Symbol(model_opt_string))
            model_opt = model_opt_func(model_stepsize)
            (gvfn=gvfn_opt, model=model_opt)
        else
            gvfn_opt
        end
    end
    # =============================================================

    # get horde
    horde = TimeSeriesUtils.get_horde(parsed)
    num_gvfs = length(horde)

    # Normalizer
    normalizer = TimeSeriesUtils.getNormalizer(parsed)

    # build model
    act = FluxUtils.get_activation(parsed["activation"])
    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        GVFR_RNN(1, horde, act; init=init_func),
        Flux.data,
        Flux.Dense(num_gvfs, num_gvfs, relu; initW=init_func),
        Flux.Dense(num_gvfs, 1; initW=init_func)
    )

    # Init observation sequence and hidden state
    obs_sequence = DataStructures.CircularBuffer{Obs_t}(τ)
    hidden_state_init = GVFN.get_initial_hidden_state(chain)

    # buffers for temporal offsets
    obs_buff, h_buff = getTemporalBuffers(horizon)

    # buffers for batches
    batch_obs, batch_h, batch_gvfn_target, batch_model_target = getNewBatch()


    TimeSeriesAgent(lu,
                    opt,
                    chain,
                    normalizer,

                    obs_sequence,
                    hidden_state_init,

                    h_buff,
                    obs_buff,

                    batch_h,
                    batch_obs,
                    batch_gvfn_target,
                    batch_model_target,

                    horizon,
                    batchsize,
                    model_clip_coeff)
end

function TimeSeriesOriginalRNNAgent(parsed; rng=Random.GLOBAL_RNG)
    # RNN architecture originally used, with RNN -> linear output

    nhidden = parsed["rnn_nhidden"]
    cell = getproperty(Flux, Symbol(parsed["rnn_cell"]))

    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        cell(1, nhidden; init=init_func),
        Flux.Dense(parsed["rnn_nhidden"], 1 ; initW=init_func)
    )
    return _TimeSeriesRNNAgent(parsed, chain; rng=rng)
end

function TimeSeriesRNNAgent(parsed; rng=Random.GLOBAL_RNG)
    # Uses an architecture more similar to the GVFN, with
    # a recurrent layer producing a representation, and
    # a FC NN producing timeseries predictions from this.

    nhidden = parsed["rnn_nhidden"]
    cell = getproperty(Flux, Symbol(parsed["rnn_cell"]))
    act = FluxUtils.get_activation(parsed["activation"])

    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        cell(1, nhidden, act; init =init_func),
        Flux.Dense(nhidden, nhidden, relu; initW=init_func),
        Flux.Dense(nhidden, 1; initW=init_func)
    )
    return _TimeSeriesRNNAgent(parsed, chain; rng=rng)
end

function _TimeSeriesRNNAgent(parsed, chain; rng=Random.GLOBAL_RNG)

    # hyperparameters
    alg_string = parsed["update_fn"]
    horizon=parsed["horizon"]
    batchsize = parsed["batchsize"]
    τ=parsed["rnn_tau"]
    clip_coeff = Float32(parsed["model_clip_coeff"])

    lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = lu_func()

    # get normalizer
    normalizer = TimeSeriesUtils.getNormalizer(parsed)

    # get optimizer
    lr, β1, β2 = map(k->parsed[k], ["rnn_lr","rnn_beta1","rnn_beta2"])
    opt = getproperty(Flux, Symbol(parsed["rnn_opt"]))(lr, (β1, β2))

    obs_sequence = DataStructures.CircularBuffer{Obs_t}(τ)
    hidden_state_init = GVFN.get_initial_hidden_state(chain)

    # buffers for temporal offsets
    obs_buff, h_buff = getTemporalBuffers(horizon)

    # buffers for batches
    batch_obs, batch_h, batch_gvfn_target, batch_model_target = getNewBatch()

    TimeSeriesAgent(lu,
                    opt,
                    chain,
                    normalizer,

                    obs_sequence,
                    hidden_state_init,

                    h_buff,
                    obs_buff,

                    batch_h,
                    batch_obs,
                    batch_gvfn_target,
                    batch_model_target,

                    horizon,
                    batchsize,
                    clip_coeff)

end

function getNewBatch()
    # get empty batch buffers
    batch_obs = Vector{Obs_t}[]
    batch_h  = Hidden_t[]
    batch_gvfn_target = Vector{Float32}[]
    batch_model_target = Vector{Float32}[]
    return batch_obs, batch_h, batch_gvfn_target, batch_model_target
end

function resetBatch!(agent::TimeSeriesAgent)
    # reset the agent's batch buffers
    agent.batch_obs, agent.batch_h, agent.batch_gvfn_target, agent.batch_model_target = getNewBatch()
end

function getTemporalBuffers(horizon::Int)
    h_buff = DataStructures.CircularBuffer{Hidden_t}(horizon)
    obs_buff = DataStructures.CircularBuffer{Vector{Obs_t}}(horizon)
    return obs_buff, h_buff
end

function MinimalRLCore.start!(agent::TimeSeriesAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    # init observation sequence
    fill!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # init hidden state
    agent.hidden_state_init = get_initial_hidden_state(agent.chain)
end


function MinimalRLCore.step!(agent::TimeSeriesAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)

    # Update state seq
    push!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # copy state sequence/hidden state into temporal offset buffers
    push!(agent.obs_buff, copy(agent.obs_sequence))
    push!(agent.h_buff, copy(agent.hidden_state_init))

    # Update =====================================================
    if DataStructures.isfull(agent.obs_buff)

        if contains_gvfn(agent.chain)
            # compute and buffer the targets for the GVFN layer
            reset!(agent.chain, agent.hidden_state_init)
            v_tp1 = agent.chain.(agent.obs_sequence)[end].data

            gvfn_idx = find_layers_with_eq(agent.chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)
            c, Γ, _ = get(agent.chain[1].cell,
                          nothing,
                          agent.obs_sequence[end],
                          nothing)
            push!(agent.batch_gvfn_target, c.+Γ.*v_tp1)
        end

        # Add target, hidden state, and observation sequence to batch
        # ---| Target = most-recent observation; obs/hidden state = earliest in the buffer
        push!(agent.batch_obs, copy(agent.obs_buff[1]))
        push!(agent.batch_h, copy(agent.h_buff[1]))
        push!(agent.batch_model_target, copy(env_s_tp1))

        if length(agent.batch_obs) == agent.batchsize
            update!(agent.chain,
                    agent.opt,
                    agent.lu,
                    agent.batchsize,
                    agent.batch_h,
                    agent.batch_obs,
                    agent.batch_gvfn_target,
                    agent.batch_model_target;
                    max_norm = agent.model_clip_coeff)


            # Reset the batch buffers
            resetBatch!(agent)
        end
    end

    # Predict ====================================================
    # Get  output/predictions
    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.obs_sequence)[end]

    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.obs_sequence[1])

    # Prediction for time t
    return out_preds.data
end

function predict!(agent::TimeSeriesAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)
    # for validation/test; predict, updating hidden states, but don't update models

    # update the hidden state
    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.obs_sequence[end])

    # Update the sequence of observations
    push!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # reset the chain's initial hidden state and run through the observation sequence
    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain(agent.obs_sequence[end])

    return out_preds.data
end


mutable struct TimeSeriesAuxTaskAgent{L, G, O, C, N, H, Φ} <: MinimalRLCore.AbstractAgent where {L<:LearningUpdate, G<:AbstractHorde}
    lu::L
    horde::G
    opt::O
    chain::C
    normalizer::N

    obs_sequence::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H

    h_buff::DataStructures.CircularBuffer{H}
    obs_buff::DataStructures.CircularBuffer{Vector{Φ}}

    batch_h::Vector{H}
    batch_obs::Vector{Vector{Φ}}
    batch_gvfn_target::Vector{Φ}
    batch_model_target::Vector{Φ}

    horizon::Int
    batchsize::Int
    model_clip_coeff::Float32
end

num_gvfs(a::TimeSeriesAuxTaskAgent) = length(a.horde)

function TimeSeriesOriginalAuxTaskAgent(parsed; rng=Random.GLOBAL_RNG)
    # RNN architecture originally used, with RNN -> linear output
    nhidden = parsed["rnn_nhidden"]
    cell = getproperty(Flux, Symbol(parsed["rnn_cell"]))

    horde = TimeSeriesUtils.get_horde(parsed)
    num_gvfs = length(horde)

    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        cell(1, nhidden; init=init_func),
        Flux.Dense(nhidden, 1+num_gvfs; initW=init_func)
    )
    return _TimeSeriesAuxTaskAgent(parsed, chain, horde; rng=rng)
end

function TimeSeriesAuxTaskAgent(parsed; rng=Random.GLOBAL_RNG)
    # Uses an architecture more similar to the GVFN, with
    # a recurrent layer producing a representation, and
    # a FC NN producing timeseries predictions from this.

    nhidden = parsed["rnn_nhidden"]
    cell = getproperty(Flux, Symbol(parsed["rnn_cell"]))
    act = FluxUtils.get_activation(parsed["activation"])

    horde = TimeSeriesUtils.get_horde(parsed)
    num_gvfs = length(horde)

    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        cell(1, nhidden; init=init_func),
        Flux.Dense(nhidden, nhidden, relu; initW=init_func),
        Flux.Dense(nhidden, 1+num_gvfs; initW=init_func)
    )
    return _TimeSeriesAuxTaskAgent(parsed, chain, horde; rng=rng)
end

function _TimeSeriesAuxTaskAgent(parsed, chain, horde; rng=Random.GLOBAL_RNG)
    # Called from an initial constructor which builds the chain/horde (above)

    # hyperparameters
    alg_string = parsed["update_fn"]
    horizon=parsed["horizon"]
    batchsize = parsed["batchsize"]
    τ=parsed["rnn_tau"]
    clip_coeff = Float32(parsed["model_clip_coeff"])

    lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = lu_func()

    # get normalizer
    normalizer = TimeSeriesUtils.getNormalizer(parsed)

    # get optimizer
    lr, β1, β2 = map(k->parsed[k], ["rnn_lr","rnn_beta1","rnn_beta2"])
    opt = getproperty(Flux, Symbol(parsed["rnn_opt"]))(lr, (β1,β2))

    obs_sequence = DataStructures.CircularBuffer{Obs_t}(τ)
    hidden_state_init = GVFN.get_initial_hidden_state(chain)

    # buffers for temporal offsets
    obs_buff, h_buff = getTemporalBuffers(horizon)

    # buffers for batches
    batch_obs, batch_h, batch_gvfn_target, batch_model_target = getNewBatch()

    TimeSeriesAuxTaskAgent(lu,
                           horde,
                           opt,
                           chain,
                           normalizer,

                           obs_sequence,
                           hidden_state_init,

                           h_buff,
                           obs_buff,

                           batch_h,
                           batch_obs,
                           batch_gvfn_target,
                           batch_model_target,

                           horizon,
                           batchsize,
                           clip_coeff)

end

function resetBatch!(agent::TimeSeriesAuxTaskAgent)
    # reset the agent's batch buffers
    agent.batch_obs, agent.batch_h, agent.batch_gvfn_target, agent.batch_model_target = getNewBatch()
end

function MinimalRLCore.start!(agent::TimeSeriesAuxTaskAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    # init observation sequence
    fill!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # init hidden state
    agent.hidden_state_init = get_initial_hidden_state(agent.chain)
end


function MinimalRLCore.step!(agent::TimeSeriesAuxTaskAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)

    # Update state seq
    push!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # copy state sequence/hidden state into temporal offset buffers
    push!(agent.obs_buff, copy(agent.obs_sequence))
    push!(agent.h_buff, copy(agent.hidden_state_init))

    # Update =====================================================
    if DataStructures.isfull(agent.obs_buff)

        # compute and buffer the targets for the GVFN layer
        #reset!(agent.chain, agent.hidden_state_init)
        v_tp1 = agent.chain.(agent.obs_sequence)[end].data[2:end]

        c, Γ, _ = get(agent.horde,
                      nothing,
                      agent.obs_sequence[end],
                      nothing)
        push!(agent.batch_gvfn_target, c.+Γ.*v_tp1)

        # Add target, hidden state, and observation sequence to batch
        # ---| Target = most-recent observation; obs/hidden state = earliest in the buffer
        push!(agent.batch_obs, copy(agent.obs_buff[1]))
        push!(agent.batch_h, copy(agent.h_buff[1]))
        push!(agent.batch_model_target, copy(env_s_tp1))

        if length(agent.batch_obs) == agent.batchsize
            update!(agent.chain,
                    agent.horde,
                    agent.opt,
                    agent.lu,
                    agent.batchsize,
                    agent.batch_h,
                    agent.batch_obs,
                    agent.batch_gvfn_target,
                    agent.batch_model_target;
                    max_norm = agent.model_clip_coeff)


            # Reset the batch buffers
            resetBatch!(agent)
        end
    end

    # Predict ====================================================
    # Get  output/predictions
    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.obs_sequence)[end]

    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.obs_sequence[1])

    # Prediction for time t
    return out_preds.data[1]
end

function predict!(agent::TimeSeriesAuxTaskAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)
    # for validation/test; predict, updating hidden states, but don't update models

    # update the hidden state
    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.obs_sequence[end])

    # Update the sequence of observations
    push!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # reset the chain's initial hidden state and run through the observation sequence
    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain(agent.obs_sequence[end])

    return out_preds.data[1]
end

