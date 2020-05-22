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
    batch_gvfn_target::Vector{Φ}
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

# =============================
# --- TIMESERIES FLUX AGENT ---
# =============================

mutable struct TimeSeriesFluxAgent{LU<:LearningUpdate, O, C, N, H, Φ, G} <: MinimalRLCore.AbstractAgent
    lu::LU
    opt::O
    chain::C
    normalizer::N

    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ

    h_buff::DataStructures.CircularBuffer{H}
    obs_buff::DataStructures.CircularBuffer{Vector{Φ}}

    horde::Horde{G}
    horizon::Int
end

function TimeSeriesFluxAgent(parsed, chain; rng=Random.GLOBAL_RNG)
    # ==========================================
    # hyperparameters
    # ==========================================
    alg_string = parsed["update_fn"]
    horizon = Int(parsed["horizon"])

    τ=parsed["tau"]
    gvfn_opt_string = parsed["opt"]
    gvfn_stepsize = parsed["stepsize"]
    # ==========================================


    # =============================================================
    # Instantiations
    # =============================================================

    # GVFN learning update
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func()

    # Optimizer
    gvfn_opt_func = getproperty(Flux, Symbol(gvfn_opt_string))
    opt = gvfn_opt_func(gvfn_stepsize)
    # =============================================================

    # Normalizer
    normalizer = TimeSeriesUtils.getNormalizer(parsed)

    out_horde = Horde([GVF(FeatureCumulant(1), ConstantDiscount(0.0f0), NullPolicy())])

    # Init observation sequence and hidden state
    state_list, init_state = DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1), zeros(Float32, 1)
    hidden_state_init = get_initial_hidden_state(chain)

    # buffers for temporal offsets
    obs_buff, h_buff = getTemporalBuffers(horizon)

    TimeSeriesFluxAgent(lu,
                        opt,
                        chain,
                        normalizer,

                        state_list,
                        hidden_state_init,
                        init_state,

                        h_buff,
                        obs_buff,

                        out_horde,
                        horizon)
end

TimeSeriesFluxGVFNAgent(parsed; rng=Random.GLOBAL_RNG) = TimeSeriesFluxAgent(parsed, GVFNChain(parsed; rng=rng))
TimeSeriesFluxRNNAgent(parsed; rng=Random.GLOBAL_RNG) = TimeSeriesFluxAgent(parsed, RNNChain(parsed; rng=rng))

function GVFNChain(parsed; rng=Random.GLOBAL_RNG)
    # get horde
    horde = TimeSeriesUtils.get_horde(parsed)
    num_gvfs = length(horde)

    # build model
    act = getproperty(Flux, Symbol(parsed["activation"]))
    init_func = (dims...)->glorot_uniform(rng, dims...)

    Flux.Chain(
        GVFR_RNN(1, horde, act; init=init_func),
        Flux.data,
        Flux.Dense(num_gvfs, num_gvfs, relu; initW=init_func),
        Flux.Dense(num_gvfs, 1; initW=init_func)
    )
end

function RNNChain(parsed; rng=Random.GLOBAL_RNG)
    cell = getproperty(Flux, Symbol(parsed["cell"]))
    init_func = (dims...)->glorot_uniform(rng, dims...)

    Flux.Chain(
        cell(1, parsed["nhidden"]; init=(dims...)->glorot_uniform(rng, dims...)),
        Flux.Dense(parsed["nhidden"], 1 ; initW=(dims...)->glorot_uniform(rng, dims...))
    )
end

build_new_feat(agent::TimeSeriesFluxAgent, state) = agent.normalizer(state)

function MinimalRLCore.start!(agent::TimeSeriesFluxAgent, env_s_tp1, rng=Random.GLOBAL_RNG)

    fill!(agent.state_list, build_new_feat(agent, env_s_tp1))

    push!(agent.state_list, build_new_feat(agent, env_s_tp1))
    agent.hidden_state_init = get_initial_hidden_state(agent.chain)
    agent.s_t =build_new_feat(agent, env_s_tp1)
end


function MinimalRLCore.step!(agent::TimeSeriesFluxAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1))

    push!(agent.obs_buff, copy(agent.state_list))
    push!(agent.h_buff, copy(agent.hidden_state_init))

    if isfull(agent.obs_buff)
        # RNN update function
        update!(agent.chain,
                agent.horde,
                agent.opt,
                agent.lu,
                agent.h_buff[1],
                agent.obs_buff[1],
                env_s_tp1,
                1,
                1.0f0)
        # End update function
    end

    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.state_list)[end]

    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.state_list[1])

    agent.s_t = build_new_feat(agent, env_s_tp1)
    return out_preds.data
end

function predict!(agent::TimeSeriesFluxAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG)
    # for validation/test; predict, updating hidden states, but don't update models

    # Update the sequence of observations
    push!(agent.state_list, build_new_feat(agent, env_s_tp1))

    # reset the chain's initial hidden state and run through the observation sequence
    reset!(agent.chain, agent.hidden_state_init)
    out_preds = agent.chain.(agent.state_list)[end]

    # update the hidden state
    agent.hidden_state_init =
        get_next_hidden_state(agent.chain, agent.hidden_state_init, agent.state_list[1])

    return out_preds.data
end
