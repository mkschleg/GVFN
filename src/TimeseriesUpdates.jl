#=
  QUARANTINED TIMESERIES CODE. ONLY USE IN TIME SERIES EXPERIMENTS.
=#

struct BatchTD <: LearningUpdate
end

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
function update!(gvfn, opt, lu::BatchTD, hidden_states, states, targets, action_t=nothing, b_prob=1.0) where {T <: AbstractGVFLayer}
    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh, gvfn.cell.b])
    N = length(hidden_states)

    δ = param(0.0)
    for t=1:N
        reset!(gvfn, hidden_states[t])
        v = gvfn(states[t])
        δ += mean(0.5*(v.-targets[t]).^2)
    end
    δ /= N

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

