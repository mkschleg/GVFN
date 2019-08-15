
using Statistics
using LinearAlgebra: Diagonal
import Flux.Tracker.update!

using Flux.Optimise: apply!

abstract type LearningUpdate end

function update!(gvfn::Flux.Recur{T}, lu::LearningUpdate, h_init, state_seq, env_state_tp1) where {T <: AbstractGVFLayer} end
function update!(gvfn::AbstractGVFLayer, lu::LearningUpdate, h_init, state_seq, env_state_tp1)
    throw("$(typeof(lu)) not implemented for $(typeof(gvfn)). Try Flux.Recur{$(typeof(gvfn))} ")
end

# Don't use TDLambda with recurrent learning...
# Assumed incremental
mutable struct TDLambda <: LearningUpdate
    λ::Float64
    traces::IdDict
    γ_t::IdDict
    TDLambda(λ) = new(λ, IdDict(), IdDict())
end

function update!(gvfn::Flux.Recur{T}, opt, lu::TDLambda, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: AbstractGVFLayer}

    λ = lu.λ
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, π_prob = get(gvfn.cell, env_state_tp1, preds_tilde)
    γ_t = get!(lu.γ_t, gvfn, zeros(Float64, size(discounts)...))::Array{Float64, 1}

    ρ = π_prob./b_prob

    targets = cumulants .+ discounts.*preds_tilde
    δ = preds_t .- targets
    grads = gradient(()->sum(δ), params(gvfn))

    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        e = get!(lu.traces, weights, zero(weights))::typeof(Flux.data(weights))
        # println(size(e))
        e .= ρ.*(e.*(γ_t.*λ) + grads[weights].data)
        Flux.Tracker.update!(opt, weights, e.*(δ))
    end

    γ_t .= discounts

    # return Flux.data.(preds)
end

function update!(gvfn::Flux.Recur{T}, opt, lu::TDLambda, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: GVFRActionLayer}

    λ = lu.λ
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    γ_t = get!(lu.γ_t, gvfn, zeros(Float64, size(discounts)...))::Array{Float64, 1}

    ρ = π_prob/b_prob

    targets = cumulants .+ discounts.*preds_tilde
    δ = preds_t .- targets
    grads = gradient(()->sum(δ), params(gvfn))
    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])
    # println(length(prms))
    for weights in prms
        e = get!(lu.traces, weights, zero(weights))::typeof(Flux.data(weights))
        e .= (e.*(γ_t.*λ)' + Flux.data(grads[weights])).*(ρ')
        Flux.Tracker.update!(opt, weights, e.*(δ'))
    end

    γ_t .= discounts

    # return preds
end

function update!(model::SingleLayer, horde::AbstractHorde, opt, lu::TDLambda, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)

    # println(state_seq)
    λ = lu.λ
    v = model.(state_seq[end-1:end])
    v_prime_t = deriv(model, state_seq[end-1])

    c, γ, π_prob = get(horde, action_t, env_state_tp1, Flux.data(v[end]))
    γ_t = get!(lu.γ_t, model, zeros(Float64, size(γ)...))::Array{Float64, 1}

    ρ = π_prob./b_prob
    δ = ρ.*tderror(v[end-1], c, γ, Flux.data(v[end]))
    Δ = δ.*v_prime_t

    e = get!(lu.traces, model.W, zero(model.W))::typeof(model.W)
    e .= (e.*(γ_t.*λ) .+ v_prime_t.*state_seq[end-1]').*(ρ)
    model.W .-= apply!(opt, model.W, e.*(δ))

    e = get!(lu.traces, model.b, zero(model.b))::typeof(model.b)
    e .= (e.*(γ_t.*λ) .+ v_prime_t).*(ρ)
    model.b .-= apply!(opt, model.b, e.*(δ))

    γ_t .= γ
end


struct TD <: LearningUpdate
end

function update!(out_model, rnn::Flux.Recur{T},
                 horde::AbstractHorde,
                 opt, lu::TD, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0; prms=nothing) where {T}

    reset!(rnn, h_init)
    rnn_out = rnn.(state_seq)
    preds = out_model.(rnn_out)
    cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Flux.data(preds[end]))

    grads = Flux.Tracker.gradient(()->δ, Flux.params(out_model, rnn))
    reset!(rnn, h_init)
    for weights in Flux.params(out_model, rnn)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end


# Update for RNNs w/ Auxiliary tasks
function update!(out_model, rnn::Flux.Recur{T},
                 horde::AbstractHorde, at_horde::AbstractHorde,
                 opt, lu::TD, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0; prms=nothing) where {T}

    reset!(rnn, h_init)
    rnn_out = rnn.(state_seq)
    preds = out_model.(rnn_out)

    preds_horde_t = preds[end-1][1:length(horde)]
    preds_horde_tp1 = Flux.data(preds[end][1:length(horde)])

    preds_at_t = preds[end-1][(1+length(horde)):end]
    preds_at_tp1 = Flux.data(preds[end][(1+length(horde)):end])
    
    cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds_horde_tp1))
    ρ = Float32.(π_prob./b_prob)
    δ_horde = offpolicy_tdloss(ρ, preds_horde_t, Float32.(cumulants), Float32.(discounts), preds_horde_tp1)

    cumulants, discounts, π_prob = get(at_horde, action_t, env_state_tp1, Flux.data(preds_at_tp1))
    ρ = Float32.(π_prob./b_prob)
    δ_at = offpolicy_tdloss(ρ, preds_at_t, Float32.(cumulants), Float32.(discounts), preds_at_tp1)

    grads = Flux.Tracker.gradient(()->(δ_horde + δ_at), Flux.params(out_model, rnn))
    reset!(rnn, h_init)
    for weights in Flux.params(out_model, rnn)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end


function update!(model, horde::AbstractHorde, opt, lu::TD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)

    if prms == nothing
        prms = params(model)
    end
    preds = model.(state_seq[end-1:end])
    c, γ, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = π_prob./b_prob
    grads = Flux.gradient(()->tdloss(preds[end-1], c, γ, Flux.data(preds[end])), prms)
    for weights in prms
        update!(opt, weights, -ρ.*grads[weights])
    end
end


function update!(model::SingleLayer, horde::AbstractHorde, opt, lu::TD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)

    v = model.(state_seq[end-1:end])
    v_prime_t = deriv(model, state_seq[end-1])

    c, γ, π_prob = get(horde, action_t, env_state_tp1, Flux.data(v[end]))
    ρ = π_prob./b_prob
    δ = ρ.*tderror(v[end-1], c, γ, Flux.data(v[end]))
    Δ = δ.*v_prime_t
    model.W .-= apply!(opt, model.W, Δ*state_seq[end-1]')
    model.b .-= apply!(opt, model.b, Δ)
end

function update!(gvfn::Flux.Recur{T}, opt, lu::TD, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: GVFRActionLayer}

    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob/b_prob

    # targets = cumulants .+ discounts.*preds_tilde
    # δ = targets .- preds_t
    grads = gradient(()->tdloss(preds_t, cumulants, discounts, preds_tilde), prms)

    for weights in prms
        Flux.Tracker.update!(opt, weights, -grads[weights].*((ρ)'))
    end
    # return preds
end

mutable struct RTD <: LearningUpdate
    # α::Float64
    RTD() = new()
end



function update!(gvfn::Flux.Recur{T}, opt, lu::RTD, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: AbstractGVFLayer}

    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])
    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob ./ b_prob

    grads = Tracker.gradient(()->offpolicy_tdloss_gvfn(Float32.(ρ), preds_t, Float32.(cumulants), Float32.(discounts), preds_tilde), prms)
    for weights in prms
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end

function update!(gvfn::Flux.Recur{T}, opt, lu::RTD, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: GVFRActionLayer}

    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])
    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob/b_prob

    # grads = Tracker.gradient(() ->tdloss(preds_t, cumulants, discounts, preds_tilde), prms)
    grads = Tracker.gradient(()->offpolicy_tdloss_gvfn(Float32.(ρ), preds_t, Float32.(cumulants), Float32.(discounts), preds_tilde), prms)
    for weights in prms
        Flux.Tracker.update!(opt, weights, grads[weights])
    end

end

function update!(model::SingleLayer, horde::AbstractHorde, opt, lu::RTD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)
    update!(model, horde, opt, TD(), state_seq, env_state_tp1, action_t, b_prob; prms=prms)
end

# function update!(gvfn::Flux.Recur{T}, horde::AbstractHorde,
#                  out_model, out_horde::AbstractHorde,
#                  opt, lu::RTD, h_init,
#                  state_seq, env_state_tp1, action_t=nothing, b_prob=1.0) where {T}

#     update!(gvfn, opt, lu, h_init, state_seq, env_s_tp1)

#     reset!(gvfn, h_init)
#     preds = gvfn.(state_seq)

#     update!(out_model, out_horde, opt, lu, Flux.data.(preds), env_s_tp1)

#     out_preds = out_model(preds[end])

#     return preds, out_preds
# end

mutable struct RTDC <: LearningUpdate
    # α::Float64
    β::Float64
    h::IdDict
    RTD(α) = new(α)
end

function update!(gvfn::Flux.Recur{T}, lu::RTDC, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    # α = lu.α

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, ρ = get(gvfn.cell, preds_tilde, env_state_tp1)
    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    prms = params(gvfn)
    # println(prms)
    grads = Tracker.gradient(() ->mean(δ.^2), prms)
    for weights in prms
        Flux.Tracker.update!(weights, -α.*grads[weights])
    end
end

mutable struct RTD_jacobian <: LearningUpdate
    # α::Float64
    J::IdDict
    RTD_jacobian() = new(IdDict{Any, Array{Float64, 3}}())
end

function update!(gvfn::Flux.Recur{T}, lu::RTD_jacobian, h_init, states, env_state_tp1, b_prob=1.0) where {T <: AbstractGVFLayer}

    # α = lu.α
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])
    action_t = states[end-1][1]

    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob/b_prob

    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    jacobian!(lu.J, δ, Params([gvfn.cell.Wx, gvfn.cell.Wh]))

    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        Flux.Tracker.update!(weights, -α.*sum(δ.*lu.J[weights]; dims=1)[1,:,:])
    end

    # return preds
end


# For timeseries experiments TIME SERIES ONLY!!!!!!
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
        δ_all[t] = mean(tderror(preds[t], Float32.(cumulants), Float32.(discounts), Flux.data(preds[t+1])).^2)
    end

    grads = Flux.Tracker.gradient(()->mean(δ_all), Flux.params(out_model, rnn))
    reset!(rnn, h_init)
    for weights in Flux.params(out_model, rnn)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end

end

# GVFN
function update!(model, horde::AbstractHorde, opt, lu::BatchTD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)
    
    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    
    # preds_t = preds[end-1]
    # preds_tilde = Flux.data(preds[end])
    δ_all = param(zeros(length(preds)-1))
    for t in 1:(length(preds)-1)
        cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, Flux.data(preds[t+1]))
        ρ = π_prob ./ b_prob
        δ_all[t] = mean(tderror(preds[t], Float32.(cumulants), Float32.(discounts), Flux.data(preds[t+1])).^2)
    end

    grads = Tracker.gradient(()->mean(δ_all), prms)
    for weights in prms
        Flux.Tracker.update!(opt, weights, grads[weights])
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

function update!(gvfn::Flux.Recur{T}, opt, lu::BatchTD, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: GVFRActionLayer}


    # return preds
end


