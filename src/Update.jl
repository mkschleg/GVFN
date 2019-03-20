
using Statistics
using LinearAlgebra: Diagonal
using Flux.Tracker: update!




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

    cumulants, discounts, π_prob = get_question_parameters(gvfn.cell, env_state_tp1, preds_tilde)
    γ_t = get!(lu.γ_t, gvfn, zeros(Float64, size(discounts)...))::Array{Float64, 1}

    ρ = π_prob/b_prob

    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t
    grads = gradient(()->sum(δ), params(gvfn))

    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        e = get!(lu.traces, weights, zeros(typeof(Flux.data(weights[1])), size(weights)...))::Array{typeof(Flux.data(weights[1])), 2}
        e .= ρ.*(convert(Array{Float64, 2}, Diagonal(γ_t)) * λ * e - grads[weights].data)
        Flux.Tracker.update!(opt, weights, e.*(δ))
    end

    γ_t .= discounts

    return Flux.data.(preds)
end

function train!(gvfn::Flux.Recur{T}, opt, lu::TDLambda, h_init, states, env_state_tp1, action_t=nothing, b_prob=1.0) where {T <: GVFRActionLayer}

    λ = lu.λ
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, π_prob = get_question_parameters(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    γ_t = get!(lu.γ_t, gvfn, zeros(Float32, size(discounts)...))::Array{Float32, 1}

    ρ = π_prob/b_prob

    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t
    grads = gradient(()->sum(δ), params(gvfn))
    prms = Params([gvfn.cell.Wx, gvfn.cell.Wh])
    # println(length(prms))
    for weights in prms
        # println("top")
        e = get!(lu.traces, weights, zeros(typeof(Flux.data(weights[1])), size(weights)...))::Array{typeof(Flux.data(weights[1])), 3}
        # for a in 1:size(weights)[1]
        #     e[a, :, :] .= (convert(Array{Float64, 2}, Diagonal(γ_t)) .* λ) * e[a, :, :]
        # end
        # e .= (e - grads[weights].data).*(ρ')
        e .= (e.*(γ_t.*λ)' - Flux.data(grads[weights])).*(ρ')
        Flux.Tracker.update!(opt, weights, e.*(δ'))
    end

    γ_t .= discounts

    return Flux.data.(preds)
end


struct TD <: AbstractUpdate
end

function train!(model, horde::AbstractHorde, opt, lu::TD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0)


    preds = model.(state_seq)

    # println(length(env_state_tp1))
    c, γ, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = π_prob./b_prob

    # println(size(c), " ", size(γ), " ", size(π_prob), " ", size(preds[end]), " ", size(preds[end-1]))

    # println(c)
    # println(γ)

    δ = preds[end-1] .- (c .+ γ.*Flux.data(preds[end]))
    grads = Flux.gradient(()->mean(δ.^2), params(model))

    for weights in params(model)
        update!(opt, weights, -ρ.*grads[weights])
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
end

mutable struct RTD <: AbstractUpdate
    # α::Float64
    RTD() = new()
end

function train!(gvfn::Flux.Recur{T}, lu::RTD, h_init, states, env_state_tp1, prms, action_prob=1.0) where {T <: AbstractGVFLayer}

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
