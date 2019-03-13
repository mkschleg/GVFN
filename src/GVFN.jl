


module GVFN
using Reexport

@reexport using JuliaRL

include("Environments.jl")

# export GVFNetwork
# include("GVFNetwork.jl")

abstract type AbstractGVFLayer end

function get(gvfn::AbstractGVFLayer, stp1, preds_tp1) end

mutable struct GVFLayer{F, A, V} <: AbstractGVFLayer
    σ::F
    Wx::A
    Wh::A
    cumulants::AbstractArray
    discounts::AbstractArray
    h::V
end

GVFLayer(num_gvfs, num_ext_features, cumulants, discounts; init=Flux.glorot_uniform, σ_int=σ) =
    GVFLayer(
        σ_int,
        param(0.1.*init(num_gvfs, num_ext_features)),
        param(0.1.*init(num_gvfs, num_gvfs)),
        cumulants,
        discounts,
        param(Flux.zeros(num_gvfs)))

function get_question_parameters(gvfn::GVFLayer{F,A,V}, preds_tilde, state_tp1) where {F, A, V}
    cumulants = [gvfn.cumulants[i](state_tp1, preds_tilde) for i in 1:length(gvfn.cumulants)]
    discounts = [gvfn.discounts[i](state_tp1) for i in 1:length(gvfn.cumulants)]
    return cumulants, discounts, ones(size(gvfn.discounts))
end

function (m::GVFLayer)(h, x)
    new_h = m.σ.(m.Wx*x + m.Wh*h)
    return new_h, new_h
end


Flux.hidden(m::GVFLayer) = m.h
Flux.@treelike GVFLayer
GVFN(args...; kwargs...) = Flux.Recur(GVFLayer(args...; kwargs...))

function reset!(m, h_init)
    Flux.reset!(m)
    m.state.data .= h_init
end

function jacobian(δ, pms)
    k  = length(δ)
    J = IdDict()
    for id in pms
        v = get!(J, id, zeros(k, size(id)...))
        for i = 1:k
            Flux.back!(δ[i], once = false) # Populate gradient accumulator
            v[i, :,:] .= id.grad
            id.grad .= 0 # Reset gradient accumulator
        end
    end
    J
end

function jacobian!(J::IdDict, δ::TrackedArray, pms::Params)
    k  = length(δ)
    for i = 1:k
        Flux.back!(δ[i], once = false) # Populate gradient accumulator
        for id in pms
            v = get!(J, id, zeros(typeof(id[1].data), k, size(id)...))::Array{typeof(id[1].data), 3}
            v[i, :, :] .= id.grad
            id.grad .= 0 # Reset gradient accumulator
        end
    end
end

abstract type Optimizer end

function train!(gvfn::Flux.Recur{T}, opt::Optimizer, h_init, state_seq, env_state_tp1) where {T <: AbstractGVFLayer} end
function train!(gvfn::AbstractGVFLayer, opt::Optimizer, h_init, state_seq, env_state_tp1)
    throw("$(typeof(opt)) not implemented for $(typeof(gvfn)). Try Flux.Recur{$(typeof(gvfn))} ")
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

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, preds_tilde, env_state_tp1)


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



end
