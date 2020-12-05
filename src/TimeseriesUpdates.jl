#=
  QUARANTINED TIMESERIES CODE. ONLY USE IN TIME SERIES EXPERIMENTS.
=#

mutable struct JankyGVFLayer{A,V}
    W::A
    H::A
    b::V
end


JankyGVFLayer(in::Integer, out::Integer; init=(dims...)->zeros(Float32, dims...)) =
    JankyGVFLayer(Flux.param(init(out, in)), Flux.param(init(out,out)), Flux.param(zeros(out)))

(layer::JankyGVFLayer)(x,h) = layer.W*x .+ layer.H*h .+ layer.b


# struct BatchTD <: LearningUpdate
# end

function update!(out_model, rnn::Flux.Recur{T},
                 horde::AbstractHorde,
                 opt, lu::BatchTD, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0; prms=nothing) where {T}

    reset!(rnn, h_init)

    rnn_out = rnn.(state_seq)
    preds = out_model.(rnn_out)

    δ_all = param(zeros(length(preds)-1))
    for t in 1:(length(preds)-1)
        cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, preds[t+1].data)
        ρ = Float64.(π_prob./b_prob)
        δ_all[t] = mean(0.5.*tderror(preds[t], Float64.(cumulants), Float64.(discounts), preds[t+1].data).^2)
    end

    grads = Flux.Tracker.gradient(()->mean(δ_all), Flux.params(out_model, rnn))
    reset!(rnn, h_init)
    for weights in Flux.params(out_model, rnn)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end

# GVFN

function update!(gvfn::JankyGVFLayer, opt, lu::BatchTD, hidden_states, states, targets, action_t=nothing, b_prob=1.0)
    prms = Params([gvfn.W, gvfn.H, gvfn.b])
    N = length(hidden_states)

    δ = param(0.0)
    for t=1:N
        v = gvfn(states[t], hidden_states[t])
        δ += mean(0.5*(v.-targets[t]).^2)
    end
    δ/=N

    grads = Tracker.gradient(()->δ, prms)
    for weights in prms
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end


function update!(model::Flux.Chain, horde::AbstractHorde, opt, lu::BatchTD, state_seq, targets, action_t=nothing, b_prob=1.0; prms=nothing)
    prms = params(model)

    v = vcat(model.(state_seq)...)
    δ = Flux.mse(v, targets)

    grads = Tracker.gradient(()->δ, prms)

    c = get_clip_coeff(grads,prms; max_norm = 0.25)
    for p in prms
        Flux.Tracker.update!(opt, p, c.*grads[p])
    end
end

function update!(model::SingleLayer, horde::AbstractHorde, opt, lu::BatchTD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)
    v = model.(state_seq)
    v_prime_t = [deriv(model, state) for state in state_seq]

    c, γ, π_prob = get(horde, action_t, env_state_tp1, v[end].data)
    ρ = π_prob./b_prob
    δ = ρ.*tderror(v[end-1], c, γ, v[end].data)
    Δ = δ.*v_prime_t
    model.W .-= apply!(opt, model.W, Δ*state_seq[end-1]')
    model.b .-= apply!(opt, model.b, Δ)
end

# Janky batch update for RNN batch n-horizon updates
function update!(chain,
                  horde::H,
                  opt,
                  lu::BatchTD,
                  batchsize::Int,
                  batch_h_init,
                  batch_state_seq,
                  batch_gvfn_target,
                  batch_model_target;
                  max_norm = 0.25) where {H<:AbstractHorde}
    action_t = nothing # Never actions in timeseries experiments
    ρ = 1.0            # No off-policy learning in the timeseries stuff
    b_prob = 1.0f0     #

    # create batches like a normal human
    kys = keys(batch_h_init[1])
    # h_init = IdDict(k=>cat([batch_h_init[i][k] for i=1:batchsize]...; dims=2) for k ∈ kys)
    h_init = make_batch_h_init(batch_h_init)
    state_seq = [cat(getindex.(batch_state_seq, t)...; dims=2) for t∈1:length(batch_state_seq[1])]
    gvfn_target = cat(batch_gvfn_target...; dims=2)
    model_target = cat(batch_model_target...; dims=2)

    # Update GVFN First
    ℒ_aux, preds = begin

        reset!(chain, h_init)
        preds = chain.(state_seq)
        0.5f0*mean((gvfn_target-preds[end][2:end,:]).^2), preds
    end
    ℒ_out = 0.5f0*mean((preds[end][1,:]' - model_target).^2)

    prms = Flux.params(chain)
    grads = Flux.Tracker.gradient(() -> (ℒ_aux + ℒ_out) * 1 // 2, prms)

    clip_coeff = FluxUtils.grad_clip_coeff(prms,grads,max_norm)
    for weights in prms
        Flux.Tracker.update!(opt, weights, grads[weights] * clip_coeff)
    end
end

function make_batch_h_init(batch_h_init)
    id_dict = IdDict()
    kys = keys(batch_h_init[1])
    batchsize = length(batch_h_init)
    for k ∈ kys
        if batch_h_init[1][k] isa Tuple
            id_dict[k] = (cat([batch_h_init[i][k][1] for i=1:batchsize]...; dims=2), cat([batch_h_init[i][k][2] for i=1:batchsize]...; dims=2))
        else
            id_dict[k] = cat([batch_h_init[i][k] for i=1:batchsize]...; dims=2)
        end
    end
    id_dict
end

function update!(chain,
                  #horde::H,
                  opt,
                  lu::BatchTD,
                  batchsize::Int,
                  batch_h_init,
                  batch_state_seq,
                  batch_gvfn_target,
                  batch_model_target;
                  max_norm = 0.25) where {H}

    action_t = nothing # Never actions in timeseries experiments
    ρ = 1.0            # No off-policy learning in the timeseries stuff
    b_prob = 1.0f0     #

    # create batches like a normal human
    kys = keys(batch_h_init[1])
    # h_init = IdDict(k=>cat([batch_h_init[i][k] for i=1:batchsize]...; dims=2) for k ∈ kys)
    h_init = make_batch_h_init(batch_h_init)
    
    state_seq = [cat(getindex.(batch_state_seq, t)...; dims=2) for t∈1:length(batch_state_seq[1])]
    gvfn_target = cat(batch_gvfn_target...; dims=2)
    model_target = cat(batch_model_target...; dims=2)

    rnn_idx = contains_gvfn(chain) ? find_layers_with_eq(chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)[1]+1 : 1

    # Update GVFN First
    ℒ_gvfn, preds = begin
        if contains_gvfn(chain)
            gvfn_idx = find_layers_with_eq(chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)
            if length(gvfn_idx) != 1
                throw("Multi-layer GVFN Not available")
            end
            ℒ, v = _gvfn_mse!(chain[1:gvfn_idx[1]],
                              lu,
                              h_init,
                              state_seq,
                              gvfn_target,
                              action_t)
            # println(v[end-1])
            ℒ, chain[gvfn_idx[1]+1:end].(Flux.data.(v))
        else
            reset!(chain, h_init)
            param(0.0f0), chain.(state_seq)
        end
    end

    ℒ_out = 0.5f0*mean((preds[end] - model_target).^2)

    grads = Flux.Tracker.gradient(()->ℒ_out + ℒ_gvfn, Flux.params(chain))
    
    prms = Flux.params(chain[rnn_idx:end])
    clip_coeff = FluxUtils.grad_clip_coeff(prms,grads,max_norm)
    for weights in prms
        Flux.Tracker.update!(opt, weights, grads[weights] * clip_coeff)
    end
end

# Used when prediction layer (GVFN) and model have separate optimizers
function update!(chain,
                 opt::NamedTuple,
                 lu::BatchTD,
                 batchsize::Int,
                 batch_h_init,
                 batch_state_seq,
                 batch_gvfn_target,
                 batch_model_target;
                 max_norm=0.25) where {H}

    action_t = nothing # Never actions in timeseries experiments
    ρ = 1.0            # No off-policy learning in the timeseries stuff
    b_prob = 1.0f0     #

    # Create batches like a normal human
    kys = keys(batch_h_init[1])
    h_init = IdDict(k=>cat([batch_h_init[i][k] for i=1:batchsize]...; dims=2) for k ∈ kys)
    state_seq = [cat(getindex.(batch_state_seq, t)...; dims=2) for t∈1:length(batch_state_seq[1])]
    gvfn_target = cat(batch_gvfn_target...; dims=2)
    model_target = cat(batch_model_target...; dims=2)

    rnn_idx = contains_gvfn(chain) ? find_layers_with_eq(chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)[1]+1 : 1

    # Update GVFN First
    ℒ_gvfn, preds = begin
        if contains_gvfn(chain)
        #if false
            gvfn_idx = find_layers_with_eq(chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)
            if length(gvfn_idx) != 1
                throw("Multi-layer GVFN Not available")
            end
            ℒ, v = _gvfn_mse!(chain[1:gvfn_idx[1]],
                              lu,
                              h_init,
                              state_seq,
                              gvfn_target,
                              action_t)
            # println(v[end-1])
            ℒ, chain[gvfn_idx[1]+1:end].(Flux.data.(v))
        else
            reset!(chain, h_init)
            param(0.0f0), chain.(state_seq)
        end
    end

    ℒ_out = 0.5f0*mean((preds[end] - model_target).^2)

    grads = Flux.Tracker.gradient(()->ℒ_out + ℒ_gvfn, Flux.params(chain))

    gvfn_idx = find_layers_with_eq(chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)
    for weights in Flux.params(chain[1:gvfn_idx[1]])
        Flux.Tracker.update!(opt.gvfn, weights, grads[weights])
    end

    prms = Flux.params(chain[rnn_idx:end])
    clip_coeff = FluxUtils.grad_clip_coeff(prms,grads,max_norm)
    for weights in prms
        Flux.Tracker.update!(opt.model, weights, grads[weights] * clip_coeff)
    end
end
