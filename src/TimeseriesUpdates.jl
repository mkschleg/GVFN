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
        cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds[t+1]))
        ρ = Float32.(π_prob./b_prob)
        δ_all[t] = mean(0.5.*tderror(preds[t], Float32.(cumulants), Float32.(discounts), Flux.data(preds[t+1])).^2)
    end

    grads = Flux.Tracker.gradient(()->mean(δ_all), Flux.params(out_model, rnn))
    reset!(rnn, h_init)
    for weights in Flux.params(out_model, rnn)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end

# GVFN
function update!(gvfn, opt, lu::BatchTD, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: AbstractGVFLayer}
    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])

    reset!(gvfn, h_init)
    preds = gvfn.(states)

    δ_all = param(0.0)
    for t in 1:(length(preds)-1)
        cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, Flux.data(preds[t+1]))
        δ_all += mean(0.5*tderror(preds[t], Float32.(cumulants), Float32.(discounts), Flux.data(preds[t+1])).^2)
    end
    δ_all /= length(preds)-1

    grads = Tracker.gradient(()->δ_all, prms)
    for weights in prms
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end


function update!(model::Flux.Chain, horde::AbstractHorde, opt, lu::BatchTD, state_seq, targets, action_t=nothing, b_prob=1.0; prms=nothing)

    prms = params(model)

    v = vcat(model.(state_seq)...)
    δ = mean(0.5*(v.-targets).^2)
    grads = Tracker.gradient(()->δ, prms)

    c = get_clip_coeff(grads,prms; max_norm = 0.25)
    for p in prms
        update!(opt, p, c.*grads[p])
    end
end

function update!(model::SingleLayer, horde::AbstractHorde, opt, lu::BatchTD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)
    v = model.(state_seq)
    v_prime_t = [deriv(model, state) for state in state_seq]

    c, γ, π_prob = get(horde, action_t, env_state_tp1, Flux.data(v[end]))
    ρ = π_prob./b_prob
    δ = ρ.*tderror(v[end-1], c, γ, Flux.data(v[end]))
    Δ = δ.*v_prime_t
    model.W .-= apply!(opt, model.W, Δ*state_seq[end-1]')
    model.b .-= apply!(opt, model.b, Δ)
end

