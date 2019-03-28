
using Statistics
using LinearAlgebra: Diagonal
using Flux.Tracker: update!

using Flux.Optimise: apply!



tderror(v_t, c, γ_tp1, ṽ_tp1) =
    (v_t .- (c .+ γ_tp1.*ṽ_tp1))

#slow for some reason...
offpolicy_tdloss(ρ_t, v_t, c, γ_tp1, ṽ_tp1) =
    sum(ρ_t.*((v_t .- (c .+ γ_tp1.*ṽ_tp1)).^2)) * 1 // length(ṽ_tp1)

tdloss(v_t, c, γ_tp1, ṽ_tp1) =
    0.5*Flux.mse(v_t, Flux.data(c .+ γ_tp1.*ṽ_tp1))


abstract type AbstractUpdate end

function train!(gvfn::Flux.Recur{T}, lu::AbstractUpdate, h_init, state_seq, env_state_tp1) where {T <: AbstractGVFLayer} end
function train!(gvfn::AbstractGVFLayer, lu::AbstractUpdate, h_init, state_seq, env_state_tp1)
    throw("$(typeof(lu)) not implemented for $(typeof(gvfn)). Try Flux.Recur{$(typeof(gvfn))} ")
end

# Don't use TDλ with recurrent learning...
# Assumed incremental
mutable struct TDLambda <: AbstractUpdate
    λ::Float64
    traces::IdDict
    γ_t::IdDict
    TDLambda(λ) = new(λ, IdDict(), IdDict())
end

function train!(gvfn::Flux.Recur{T}, opt, lu::TDLambda, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: AbstractGVFLayer}

    λ = lu.λ
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, π_prob = get(gvfn.cell, env_state_tp1, preds_tilde)
    γ_t = get!(lu.γ_t, gvfn, zeros(Float64, size(discounts)...))::Array{Float64, 1}

    ρ = π_prob./b_prob

    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t
    grads = gradient(()->sum(δ), params(gvfn))

    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        e = get!(lu.traces, weights, zero(weights))::typeof(Flux.data(weights))
        e .= ρ.*(e.*(γ_t.*λ) - grads[weights].data)
        Flux.Tracker.update!(opt, weights, e.*(δ))
    end

    γ_t .= discounts

    # return Flux.data.(preds)
end

function train!(gvfn::Flux.Recur{T}, opt, lu::TDLambda, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: GVFRActionLayer}

    λ = lu.λ
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    γ_t = get!(lu.γ_t, gvfn, zeros(Float64, size(discounts)...))::Array{Float64, 1}

    ρ = π_prob/b_prob

    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t
    grads = gradient(()->sum(δ), params(gvfn))
    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])
    # println(length(prms))
    for weights in prms
        e = get!(lu.traces, weights, zero(weights))::typeof(Flux.data(weights))
        e .= (e.*(γ_t.*λ)' .- Flux.data(grads[weights])).*(ρ')
        Flux.Tracker.update!(opt, weights, e.*(δ'))
    end

    γ_t .= discounts

    # return preds
end


struct TD <: AbstractUpdate
end



function train!(model, horde::AbstractHorde, opt, lu::TD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)

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


function train!(model::SingleLayer, horde::AbstractHorde, opt, lu::TD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)

    # println(state_seq)
    v = model.(state_seq[end-1:end])
    v_prime_t = deriv(model, state_seq[end-1])

    c, γ, π_prob = get(horde, action_t, env_state_tp1, Flux.data(v[end]))
    ρ = π_prob./b_prob
    δ = ρ.*tderror(v[end-1], c, γ, Flux.data(v[end]))
    Δ = δ.*v_prime_t
    model.W .+= -apply!(opt, model.W, Δ*state_seq[end-1]')
    model.b .+= -apply!(opt, model.b, Δ)
end

function train!(gvfn::Flux.Recur{T}, opt, lu::TD, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: GVFRActionLayer}

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

mutable struct RTD <: AbstractUpdate
    # α::Float64
    RTD() = new()
end

function train!(gvfn::Flux.Recur{T}, opt, lu::RTD, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: AbstractGVFLayer}

    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])
    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob ./ b_prob
    targets = cumulants .+ discounts.*preds_tilde
    # δ = targets .- preds_t
    grads = Tracker.gradient(()->tdloss(preds_t, cumulants, discounts, preds_tilde), prms)
    for weights in prms
        Flux.Tracker.update!(opt, weights, -ρ.*grads[weights])
    end
end

function train!(gvfn::Flux.Recur{T}, opt, lu::RTD, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: GVFRActionLayer}

    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])
    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob/b_prob
    # targets = cumulants .+ discounts.*preds_tilde
    # δ = targets .- preds_t

    grads = Tracker.gradient(() ->tdloss(preds_t, cumulants, discounts, preds_tilde), prms)
    for weights in prms
        Flux.Tracker.update!(opt, weights, -grads[weights].*(ρ'))
    end
    # return preds
end

mutable struct RTDC <: AbstractUpdate
    # α::Float64
    β::Float64
    h::IdDict
    RTD(α) = new(α)
end

function train!(gvfn::Flux.Recur{T}, lu::RTDC, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

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


mutable struct RTD_jacobian <: AbstractUpdate
    # α::Float64
    J::IdDict
    RTD_jacobian() = new(IdDict{Any, Array{Float64, 3}}())
end

function train!(gvfn::Flux.Recur{T}, lu::RTD_jacobian, h_init, states, env_state_tp1, b_prob=1.0) where {T <: AbstractGVFLayer}

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
