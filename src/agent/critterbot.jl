export CritterbotGVFNAgent, CritterbotRNNAgent, CritterbotAuxTaskAgent,
    CritterbotOriginalRNNAgent, CritterbotOriginalAuxTaskAgent,
    predict!

import Flux
import Random
import DataStructures
import MinimalRLCore

# =================
# --- FLUX AGENT ---
# =================

mutable struct CritterbotAgent{L, O, C, N, H, Φ, F} <: MinimalRLCore.AbstractAgent where {L<:LearningUpdate}
    lu::L
    opt::O
    chain::C
    normalizer::N

    obs_sequence::DataStructures.CircularBuffer{F}
    hidden_state_init::H

    h_buff::DataStructures.CircularBuffer{H}
    obs_buff::DataStructures.CircularBuffer{Vector{F}}

    batch_h::Vector{H}
    batch_obs::Vector{Vector{F}}
    batch_gvfn_target::Vector{Φ}
    batch_model_target::Vector{Φ}

    horizon::Int
    batchsize::Int
    model_clip_coeff::Float32
end

# Convenient type aliases
# Hidden_t = IdDict{Any,Any}
# Obs_t = Vector{Float32}

function CritterbotGVFNAgent(parsed; rng=Random.GLOBAL_RNG)

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
    # normalizer = if "tilings" ∈ keys(parsed)
    #     n = TimeSeriesUtils.getNormalizer(parsed)
    #     tc = GVFN.TileCoder(parsed["tilings"], parsed["tiles"], parsed["num_features"])
    #     (x)->begin
    #         n(tc(x))
    #     end
    # else
    #     TimeSeriesUtils.getNormalizer(parsed)
    # end

    # build model
    act_str = get(parsed, "activation", "sigmoid")
    act = FluxUtils.get_activation(act_str)
    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        GVFR_RNN(parsed["num_features"], horde, act; init=init_func),
        Flux.data,
        Flux.Dense(num_gvfs, num_gvfs, relu; initW=init_func),
        Flux.Dense(num_gvfs, parsed["num_targets"]; initW=init_func)
    )

    # Init observation sequence and hidden state
    obs_sequence = DataStructures.CircularBuffer{Obs_t}(τ)
    hidden_state_init = GVFN.get_initial_hidden_state(chain)

    # buffers for temporal offsets
    obs_buff, h_buff = getTemporalBuffers(horizon)

    # buffers for batches
    batch_obs, batch_h, batch_gvfn_target, batch_model_target = getNewBatch()


    CritterbotAgent(lu,
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

function CritterbotOriginalRNNAgent(parsed; rng=Random.GLOBAL_RNG)
    # RNN architecture originally used, with RNN -> linear output
    
    normalizer = if "tilings" ∈ keys(parsed)
        n = TimeSeriesUtils.getNormalizer(parsed)
        tc = GVFN.TileCoder(parsed["tilings"], parsed["tiles"], parsed["num_features"])
        parsed["num_features"] = size(tc)
        (x)->begin
            n(tc(x))
        end
    else
        TimeSeriesUtils.getNormalizer(parsed)
    end
    
    nhidden = parsed["rnn_nhidden"]
    cell = getproperty(Flux, Symbol(parsed["rnn_cell"]))

    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        cell(1, nhidden; init=init_func),
        Flux.Dense(parsed["rnn_nhidden"], 1 ; initW=init_func)
    )
    return _CritterbotRNNAgent(parsed, chain, normalizer; rng=rng)
end

function CritterbotRNNAgent(parsed; rng=Random.GLOBAL_RNG)
    # Uses an architecture more similar to the GVFN, with
    # a recurrent layer producing a representation, and
    # a FC NN producing timeseries predictions from this.

    normalizer = if "tilings" ∈ keys(parsed)
        n = TimeSeriesUtils.getNormalizer(parsed)
        tc = GVFN.TileCoder(parsed["tilings"], parsed["tiles"], parsed["num_features"])
        parsed["num_features"] = size(tc)
        (x)->begin
            tc(n(x))
        end
    else
        TimeSeriesUtils.getNormalizer(parsed)
    end
    
    nhidden = parsed["rnn_nhidden"]
    act_str = get(parsed, "activation", "tanh")
    act = FluxUtils.get_activation(act_str)

    init_func = (dims...)->glorot_uniform(rng, dims...)
    cell_name, nfeats = parsed["rnn_cell"], parsed["num_features"]
    cell_t = getproperty(Flux, Symbol(cell_name))
    cell = cell_name == "RNN" ?
        cell_t(nfeats, nhidden, act; init=init_func) :
        cell_t(nfeats, nhidden; init=init_func)

    chain = Flux.Chain(
        cell,
        Flux.Dense(nhidden, nhidden, relu; initW=init_func),
        Flux.Dense(nhidden, parsed["num_targets"]; initW=init_func)
    )
    return _CritterbotRNNAgent(parsed, chain, normalizer; rng=rng)
end

function CritterbotTCAgent(parsed; rng=Random.GLOBAL_RNG)
    
    n = TimeSeriesUtils.getNormalizer(parsed)
    tc = GVFN.TileCoder(parsed["tilings"], parsed["tiles"], parsed["num_features"])
    parsed["num_features"] = size(tc)
    normalizer = (x)->begin
        tc(n(x))
    end

    chain = Flux.Chain(
        TCLayer(parsed["num_features"], parsed["num_targets"])
    )
    return _CritterbotRNNAgent(parsed, chain, normalizer; rng=rng)
    
end


function CritterbotFCAgent(parsed; rng=Random.GLOBAL_RNG)
    
    # n = TimeSeriesUtils.getNormalizer(parsed)
    # tc = GVFN.TileCoder(parsed["tilings"], parsed["tiles"], parsed["num_features"])
    # parsed["num_features"] = size(tc)
    # normalizer = (x)->begin
    #     tc(n(x))
    # end
    normalizer = if "tilings" ∈ keys(parsed)
        n = TimeSeriesUtils.getNormalizer(parsed)
        tc = GVFN.TileCoder(parsed["tilings"], parsed["tiles"], parsed["num_features"])
        parsed["num_features"] = size(tc)
        (x)->begin
            tc(n(x))
        end
    else
        TimeSeriesUtils.getNormalizer(parsed)
    end
    
    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = if "network_arch" ∈ keys(parsed)
        if length(parsed["network_arch"]) == 1
            Flux.Chain(
                Dense(parsed["num_features"], parsed["network_arch"][1]["size"], FluxUtils.get_activation(parsed["network_arch"][1]["act"]); initW=init_func),
                Dense(parsed["network_arch"][1]["size"], parsed["num_targets"])
            )
        else
            Flux.Chain(
                Dense(parsed["num_features"], parsed["network_arch"][1]["size"], FluxUtils.get_activation(parsed["network_arch"][1]["act"]); initW=init_func),
                [Dense(parsed["network_arch"][i-1]["size"], parsed["network_arch"][i]["size"], FluxUtils.get_activation(parsed["network_arch"][i]["act"]); initW=init_func) for i ∈ 2:length(parsed["network_arch"])]...,
                Dense(parsed["network_arch"][end]["size"], parsed["num_targets"])
            )
        end
    else
        Flux.Chain(
            Dense(parsed["num_features"], parsed["num_targets"])
        )
    end
    return _CritterbotRNNAgent(parsed, chain, normalizer; rng=rng)
    
end

Flux.Descent(α::Float64, ::Tuple{Float64,Float64}) = Flux.Descent(α)

function _CritterbotRNNAgent(parsed, chain, normalizer; rng=Random.GLOBAL_RNG)

    # hyperparameters
    alg_string = parsed["update_fn"]
    horizon=parsed["horizon"]
    batchsize = parsed["batchsize"]
    τ=parsed["rnn_tau"]
    clip_coeff = Float32(parsed["model_clip_coeff"])

    lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = lu_func()

    # get normalizer
    # normalizer = TimeSeriesUtils.getNormalizer(parsed)

    # get optimizer
    lr, β1, β2 = map(k->parsed[k], ["rnn_lr","rnn_beta1","rnn_beta2"])
    opt = getproperty(Flux, Symbol(parsed["rnn_opt"]))(lr, (β1, β2))

    obs_sequence = if "tilings" ∈ keys(parsed)
        DataStructures.CircularBuffer{Vector{Int}}(τ)
    else
        DataStructures.CircularBuffer{Obs_t}(τ)
    end
    hidden_state_init = GVFN.get_initial_hidden_state(chain)

    # buffers for temporal offsets
    obs_buff, h_buff = getTemporalBuffers(horizon, "tilings" ∈ keys(parsed))

    # buffers for batches
    batch_obs, batch_h, batch_gvfn_target, batch_model_target = getNewBatch("tilings" ∈ keys(parsed))

    CritterbotAgent(lu,
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

function getNewBatch(tc)
    # get empty batch buffers
    batch_obs = tc ? Vector{TC_Obs_t}[] : Vector{Obs_t}[]
    batch_h  = Hidden_t[]
    batch_gvfn_target = Vector{Float32}[]
    batch_model_target = Vector{Float32}[]
    return batch_obs, batch_h, batch_gvfn_target, batch_model_target
end

function resetBatch!(agent::CritterbotAgent)
    # reset the agent's batch buffers
    agent.batch_obs, agent.batch_h, agent.batch_gvfn_target, agent.batch_model_target = getNewBatch()
end

function getTemporalBuffers(horizon::Int, tc)
    h_buff = DataStructures.CircularBuffer{Hidden_t}(horizon)
    obs_buff = tc ? DataStructures.CircularBuffer{Vector{TC_Obs_t}}(horizon) : DataStructures.CircularBuffer{Vector{Obs_t}}(horizon)
    return obs_buff, h_buff
end

function MinimalRLCore.start!(agent::CritterbotAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    # init observation sequence
    fill!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # init hidden state
    agent.hidden_state_init = get_initial_hidden_state(agent.chain)
end


function MinimalRLCore.step!(agent::CritterbotAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)

    # Update state seq
    push!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # copy state sequence/hidden state into temporal offset buffers
    # println(typeof(agent.obs_buff), typeof(agent.obs_sequence))
    # for obs ∈ agent.obs_sequence
    push!(agent.obs_buff, copy(agent.obs_sequence))
    # end
    push!(agent.h_buff, copy(agent.hidden_state_init))

    # Update =====================================================
    if DataStructures.isfull(agent.obs_buff)

        if contains_gvfn(agent.chain)
            # compute and buffer the targets for the GVFN layer
            reset!(agent.chain, agent.hidden_state_init)
            v_tp1 = agent.chain[1].(agent.obs_sequence)[end].data

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
        push!(agent.batch_model_target, copy(r))

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

function predict!(agent::CritterbotAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)
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


mutable struct CritterbotAuxTaskAgent{L, G, O, C, N, H, Φ} <: MinimalRLCore.AbstractAgent where {L<:LearningUpdate, G<:AbstractHorde}
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

num_gvfs(a::CritterbotAuxTaskAgent) = length(a.horde)

function CritterbotOriginalAuxTaskAgent(parsed; rng=Random.GLOBAL_RNG)
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
    return _CritterbotAuxTaskAgent(parsed, chain, horde; rng=rng)
end

function CritterbotAuxTaskAgent(parsed; rng=Random.GLOBAL_RNG)
    # Uses an architecture more similar to the GVFN, with
    # a recurrent layer producing a representation, and
    # a FC NN producing timeseries predictions from this.

    nhidden = parsed["rnn_nhidden"]
    cell = getproperty(Flux, Symbol(parsed["rnn_cell"]))
    act_str = get(parsed, "activation", "tanh")
    act = FluxUtils.get_activation(act_str)

    horde = TimeSeriesUtils.get_horde(parsed)
    num_gvfs = length(horde)

    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        cell(1, nhidden; init=init_func),
        Flux.Dense(nhidden, nhidden, relu; initW=init_func),
        Flux.Dense(nhidden, 1+num_gvfs; initW=init_func)
    )
    return _CritterbotAuxTaskAgent(parsed, chain, horde; rng=rng)
end

function _CritterbotAuxTaskAgent(parsed, chain, horde; rng=Random.GLOBAL_RNG)
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

    CritterbotAuxTaskAgent(lu,
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

function resetBatch!(agent::CritterbotAuxTaskAgent)
    # reset the agent's batch buffers
    agent.batch_obs, agent.batch_h, agent.batch_gvfn_target, agent.batch_model_target = getNewBatch()
end

function MinimalRLCore.start!(agent::CritterbotAuxTaskAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    # init observation sequence
    fill!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # init hidden state
    agent.hidden_state_init = get_initial_hidden_state(agent.chain)
end


function MinimalRLCore.step!(agent::CritterbotAuxTaskAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)

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
        push!(agent.batch_model_target, copy(r))

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

function predict!(agent::CritterbotAuxTaskAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)
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

