
using Statistics
using LinearAlgebra: Diagonal

abstract type Optimizer end

function train!(gvfn::Flux.Recur{T}, opt::Optimizer, h_init, state_seq, env_state_tp1) where {T <: AbstractGVFLayer} end
function train!(gvfn::AbstractGVFLayer, opt::Optimizer, h_init, state_seq, env_state_tp1)
    throw("$(typeof(opt)) not implemented for $(typeof(gvfn)). Try Flux.Recur{$(typeof(gvfn))} ")
end

# Don't use TDλ with recurrent learning...
mutable struct TDλ <: Optimizer
    α::Float64
    λ::Float64
    traces::IdDict
    TDλ(α, λ) = new(α, λ, IdDict())
end

function train!(gvfn::Flux.Recur{T}, opt::TDλ, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    α = opt.α
    λ = opt.λ
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, env_state_tp1, preds_tilde)

    # println("Cumulants: ", cumulants, " Discounts: ", discounts)

    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    # jacobian!(opt.J, δ, Params([gvfn.cell.Wx, gvfn.cell.Wh]))
    grads = gradient(()->sum(δ), params(gvfn))
    
    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        e = get!(opt.traces, weights, zeros(typeof(Flux.data(weights[1])), size(weights)...))::Array{typeof(Flux.data(weights[1])), 2}
        e .= convert(Array{Float64, 2}, Diagonal(discounts)) * λ * e - grads[weights].data
        Flux.Tracker.update!(weights, α.*e.*(δ))
    end
end

mutable struct RTD_jacobian <: Optimizer
    α::Float64
    J::IdDict
    RTD_jacobian(α) = new(α, IdDict{Any, Array{Float64, 3}}())
end

function train!(gvfn::Flux.Recur{T}, opt::RTD_jacobian, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    α = opt.α
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, env_state_tp1, preds_tilde)


    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    jacobian!(opt.J, δ, Params([gvfn.cell.Wx, gvfn.cell.Wh]))

    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        Flux.Tracker.update!(weights, -α.*sum(δ.*opt.J[weights]; dims=1)[1,:,:])
    end
end

mutable struct RTD <: Optimizer
    α::Float64
    RTD(α) = new(α)
end

function train!(gvfn::Flux.Recur{T}, opt::RTD, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    α = opt.α

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


mutable struct RTDC <: Optimizer
    α::Float64
    β::Float64
    h::IdDict
    RTD(α) = new(α)
end

function train!(gvfn::Flux.Recur{T}, opt::RTDC, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    α = opt.α

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
