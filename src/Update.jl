
using Statistics
using LinearAlgebra: Diagonal
using Flux.Tracker: update!



abstract type AbstractUpdate end

function train!(gvfn::Flux.Recur{T}, lu::AbstractUpdate, h_init, state_seq, env_state_tp1) where {T <: AbstractGVFLayer} end
function train!(gvfn::AbstractGVFLayer, lu::AbstractUpdate, h_init, state_seq, env_state_tp1)
    throw("$(typeof(lu)) not implemented for $(typeof(gvfn)). Try Flux.Recur{$(typeof(gvfn))} ")
end

# Don't use TDλ with recurrent learning...
mutable struct TDLambda <: AbstractUpdate
    λ::Float64
    traces::IdDict
    TDLambda(λ) = new(λ, IdDict())
end

function train!(gvfn::Flux.Recur{T}, opt, lu::TDLambda, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    λ = lu.λ
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, env_state_tp1, preds_tilde)

    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t
    grads = gradient(()->sum(δ), params(gvfn))

    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        e = get!(lu.traces, weights, zeros(typeof(Flux.data(weights[1])), size(weights)...))::Array{typeof(Flux.data(weights[1])), 2}
        e .= convert(Array{Float64, 2}, Diagonal(discounts)) * λ * e - grads[weights].data
        Flux.Tracker.update!(opt, weights, e.*(δ))
    end
end


struct TD <: AbstractUpdate
end

function train!(model, horde::AbstractHorde, opt, lu::TD, state_seq, env_state_tp1, hidden_init=nothing)

    if hidden_init != nothing
        reset!(model[1], hidden_init)
    end
    preds = model.(state_seq)

    c, γ, π_prob = get(horde, env_state_tp1, Flux.data(preds[end]))

    δ = preds[end-1] - (c + γ.*Flux.data(preds[end]))
    grads = Flux.gradient(()->mean(δ.^2), params(model))

    for weights in params(model)
        update!(opt, weights, -grads[weights])
    end

end

mutable struct RTD_jacobian <: AbstractUpdate
    # α::Float64
    J::IdDict
    RTD_jacobian() = new(IdDict{Any, Array{Float64, 3}}())
end

function train!(gvfn::Flux.Recur{T}, lu::RTD_jacobian, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    # α = lu.α
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, env_state_tp1, preds_tilde)


    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    jacobian!(lu.J, δ, Params([gvfn.cell.Wx, gvfn.cell.Wh]))

    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        Flux.Tracker.update!(weights, -α.*sum(δ.*lu.J[weights]; dims=1)[1,:,:])
    end
end

mutable struct RTD <: AbstractUpdate
    # α::Float64
    RTD() = new()
end

function train!(gvfn::Flux.Recur{T}, lu::RTD, h_init, states, env_state_tp1, prms) where {T <: AbstractGVFLayer}

    # α = lu.α

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])
    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, env_state_tp1, preds_tilde)
    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    prms = params(gvfn)
    grads = Tracker.gradient(() ->mean(δ.^2), prms)
    for weights in prms
        Flux.Tracker.update!(weights, -α.*grads[weights])
    end
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

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, preds_tilde, env_state_tp1)
    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    prms = params(gvfn)
    # println(prms)
    grads = Tracker.gradient(() ->mean(δ.^2), prms)
    for weights in prms
        Flux.Tracker.update!(weights, -α.*grads[weights])
    end
end
