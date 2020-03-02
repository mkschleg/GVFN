export TimeSeriesGVFNAgent, TimeSeriesRNNAgent, predict!

import Flux
import Random
import DataStructures
import MinimalRLCore

# ==================
# --- GVFN AGENT ---
# ==================

mutable struct OriginalTimeSeriesAgent{GVFNOpt,ModelOpt, J, H, Φ, M, G1, G2, N} <: MinimalRLCore.AbstractAgent
    lu::LearningUpdate
    gvfn_opt::GVFNOpt
    model_opt::ModelOpt
    gvfn::J
    normalizer::N

    batch_phi::Vector{Φ}
    batch_target::Vector{Φ}
    batch_hidden::Vector{Φ}
    batch_h::Vector{Φ}
    batch_obs::Vector{Float32}

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


function OriginalTimeSeriesAgent(parsed; rng=Random.GLOBAL_RNG)

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
    batch_phi = Vector{Float32}[]
    batch_target = Vector{Float32}[]
    batch_hidden = Vector{Float32}[]
    hidden_states = Vector{Float32}[]

    hidden_state_init = zeros(Float32, num_gvfs)

    batch_obs = Float32[]
    batch_h = Array{Float32,1}[]

    horizon = Int(parsed["horizon"])

    return OriginalTimeSeriesAgent(lu, gvfn_opt, model_opt, gvfn, normalizer, batch_phi, batch_target, batch_hidden, batch_h, batch_obs, hidden_states, hidden_state_init, zeros(Float32, 1), model, horde, out_horde, horizon, 0, batchsize)
end

function MinimalRLCore.start!(agent::OriginalTimeSeriesAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    stp1 = agent.normalizer(env_s_tp1)

    agent.h .= zero(agent.h)
    agent.h .= agent.gvfn(stp1, agent.h).data
    agent.s_t .= stp1

    agent.step+=1
end

function MinimalRLCore.step!(agent::OriginalTimeSeriesAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)
    push!(agent.hidden_states, copy(agent.h))

    if agent.step>=agent.horizon
        push!(agent.batch_h, popfirst!(agent.hidden_states))
        push!(agent.batch_obs, env_s_tp1[1])
        if length(agent.batch_obs) == agent.batchsize
            update!(agent.model, agent.out_horde, agent.model_opt, agent.lu, agent.batch_h, agent.batch_obs)

            agent.batch_obs = Float32[]
            agent.batch_h = Vector{Float32}[]
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

        agent.batch_phi = Vector{Float32}[]
        agent.batch_hidden = Vector{Float32}[]
        agent.batch_target = Vector{Float32}[]
    end

    agent.s_t .= stp1
    agent.h .= v_tp1
    agent.step+=1

    return agent.model(v_tp1).data
end

function predict!(agent::OriginalTimeSeriesAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)
    # for validation/test; predict, updating hidden states, but don't update models

    agent.h .= agent.gvfn(env_s_tp1, agent.h).data
    return agent.model(agent.h).data
end

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
    batch_target::Vector{Φ}

    horizon::Int
    batchsize::Int
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
    act = getproperty(Flux, Symbol(parsed["activation"]))
    init_func = (dims...)->glorot_uniform(rng, dims...)
    chain = Flux.Chain(
        GVFR_RNN(1, horde, act; init=init_func),
        Flux.data,
        Flux.Dense(num_gvfs, num_gvfs, relu; initW=init_func),
        Flux.Dense(num_gvfs, 1; initW=init_func)
    )

    # Init observation sequence and hidden state
    obs_sequence = DataStructures.CircularBuffer{Obs_t}(τ+1)
    hidden_state_init = GVFN.get_initial_hidden_state(chain)

    # buffers for temporal offsets
    obs_buff, h_buff = getTemporalBuffers(horizon)

    # buffers for batches
    batch_obs, batch_h, batch_target = getNewBatch()


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
                    batch_target,

                    horizon,
                    batchsize)
end

function TimeSeriesRNNAgent(parsed; rng=Random.GLOBAL_RNG)

    # hyperparameters
    alg_string = parsed["update_fn"]
    horizon=parsed["horizon"]
    batchsize = parsed["batchsize"]
    nhidden=parsed["rnn_nhidden"]
    τ=parsed["rnn_tau"]
    lr = parsed["rnn_lr"]

    lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = lu_func()

    # get normalizer
    normalizer = TimeSeriesUtils.getNormalizer(parsed)

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
    obs_buff, h_buff = getTemporalBuffers(horizon)

    # buffers for batches
    batch_obs, batch_h, batch_target = getNewBatch()

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
                    batch_target,

                    horizon,
                    batchsize)

end

function getNewBatch()
    # get empty batch buffers
    batch_obs = Vector{Obs_t}[]
    batch_h  = Hidden_t[]
    batch_target = Obs_t[]
    return batch_obs, batch_h, batch_target
end

function resetBatch!(agent::TimeSeriesAgent)
    # reset the agent's batch buffers
    agent.batch_obs, agent.batch_h, agent.batch_target = getNewBatch()
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

    # Update the sequence of observations
    push!(agent.obs_sequence, agent.normalizer(env_s_tp1))

    # reset the chain's initial hidden state and run through the observation sequence
    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.obs_sequence)[end]

    # update the hidden state
    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.obs_sequence[1])

    return out_preds.data
end
