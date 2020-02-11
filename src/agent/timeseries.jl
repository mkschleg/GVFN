export TimeSeriesAgent, TimeSeriesRNNAgent, predict!

import Flux
import Random
import DataStructures

mutable struct TimeSeriesFluxAgent{O1, O2, C, F, H, Φ} <: JuliaRL.AbstractAgent
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

function JuliaRL.start!(agent::TimeSeriesFluxAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)


    agent.s_t .= agent.build_features(env_s_tp1)
    
    fill!(agent.state_list, agent.build_features(env_s_tp1))
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    
    agent.step+=1
end

function JuliaRL.step!(agent::TimeSeriesFluxAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

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
    
    #Update Forecasting
    # if agent.step > horizon && ((agent.step-horizon) % batchsize) == 0

    #     # println(agent.step)
        
    #     prms = params(model)
        
    #     reset!(model, agent.hidden_state_init)
    #     # hs_preds = vcat(model.(agent.state_list[2:batchsize])...)
    #     hs_preds = get_hidden_states_and_preds(model, agent.state_list)
    #     preds = getindex.(hs_preds, 2)

    #     targets = hcat(agent.state_list[horizon+2:end]...)

    #     δ = Flux.mse(hcat(preds[3:batchsize+2]...), targets)

    #     grads = Tracker.gradient(()->δ, prms)

    #     clip = get_clip_coeff(grads, prms; max_norm = 0.25)
    #     for p in prms
    #         Flux.Tracker.update!(agent.model_opt, p, clip.*grads[p])
    #     end
    # # end


    # # # Update GVFN
    # # if (agent.step % batchsize) == 0

    #     # reset!(model, agent.hidden_state_init)
    #     # hidden_states = GVFN.get_hidden_states(model, agent.state_list[1:end])
    #     hidden_states = getindex.(hs_preds, 1)
    #     # println(hidden_states[end])
    #     gvfn = model[1]
        
    #     get_pred = (hidden_state, state_seq) -> begin
    #         reset!(gvfn, hidden_state)
    #         gvfn.(state_seq)[end]
    #     end

    #     preds = TrackedArray[]
    #     for i ∈ 1:(batchsize+1)
    #         push!(preds, get_pred(hidden_states[i+horizon-1], agent.state_list[i+horizon:i+horizon]))
    #     end


    #     preds_t = hcat(preds[1:end-1]...)
    #     preds_tp1 = hcat(Flux.data.(preds[2:end])...)
    #     # println(preds_t)
    #     # targets = zeros(Float32, length(preds), length(preds[1]))
    #     c = zeros(Float32, length(gvfn.cell.horde), batchsize)
    #     γ = zeros(Float32, length(gvfn.cell.horde), batchsize)
    #     for i ∈ batchsize
    #         c[:, i], γ[:, i], _ = get(gvfn.cell.horde, nothing, agent.state_list[i+horizon+1], preds_tp1[:, i])
    #     end
        
    #     ℒ = mean(mean(0.5.*(preds_t .- (c .+ γ.*preds_tp1)).^2; dims=1); dims=2)[1]

    #     grads = Tracker.gradient(()->ℒ, params(gvfn))
    #     for weights in params(gvfn)
    #         Flux.Tracker.update!(agent.gvfn_opt, weights, grads[weights])
    #     end
        
    # end
    
    
    # stp1 = agent.normalizer(env_s_tp1)
    # v_tp1 = agent.gvfn(stp1,agent.h).data
    # c, Γ, _ = get(agent.horde, nothing, env_s_tp1, v_tp1)
    # push!(agent.batch_target, c .+ Γ.*v_tp1)
    # push!(agent.batch_phi, copy(agent.s_t))
    # push!(agent.batch_hidden, copy(agent.h))
    # if length(agent.batch_phi) == agent.batchsize
    #     update!(agent.gvfn, agent.gvfn_opt, agent.lu, agent.batch_hidden, agent.batch_phi, agent.batch_target)

    #     agent.batch_phi = Vector{Float64}[]
    #     agent.batch_hidden = Vector{Float64}[]
    #     agent.batch_target = Vector{Float64}[]
    # end
    
    
    agent.s_t .= agent.build_features(env_s_tp1)
    agent.step += 1
    reset!(agent.model, agent.hidden_state_init)
    pred = agent.model.(agent.state_list)[end].data
    
    # agent.hidden_state_init = GVFN.get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1])

    return pred
end

function predict!(agent::TimeSeriesFluxAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG,kwargs...)
    # for validation/test; predict, updating hidden states, but don't update models

    push!(agent.state_list, agent.build_features(env_s_tp1))
    return agent.model.(agent.state_list)[end].data

end

mutable struct TimeSeriesAgent{GVFNOpt,ModelOpt, J, H, Φ, M, G1, G2, N} <: JuliaRL.AbstractAgent
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

function JuliaRL.start!(agent::TimeSeriesAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    stp1 = agent.normalizer(env_s_tp1)

    agent.h .= zero(agent.h)
    agent.h .= agent.gvfn(stp1, agent.h).data
    agent.s_t .= stp1

    agent.step+=1
end

function JuliaRL.step!(agent::TimeSeriesAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)
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
