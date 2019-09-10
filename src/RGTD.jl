

struct Prediction end
struct Derivative end
struct DoubleDerivative end

sigmoid(x, type::Type{Prediction}) = sigmoid(x)
sigmoid(x, type::Type{Derivative}) = sigmoid′(x)
sigmoid(x, type::Type{DoubleDerivative}) = sigmoid′′(x)

mutable struct GradientGVFN{H<:AbstractHorde, F<:Function}
    
    # Weights
    θ::Matrix{Float32}
    # Secondary Weights
    h::Matrix{Float32}

    # updates
    Δθ::Matrix{Float32}
    Δh::Matrix{Float32}
    
    # Psi
    Ψ::Matrix{Float32}
    # Phi
    Φ::Vector{Matrix{Float32}}
    # PhiPrime
    Φ′::Vector{Matrix{Float32}}
    # H
    H::Vector{Matrix{Float32}}
    # deltaH
    ΔH::Vector{Matrix{Float32}}

    # Questions Graph (𝒳)
    # GVFs
    horde::H

    # Activation
    σ::F
end

GradientGVFN(in, horde, σ; initθ=Flux.glorot_uniform) =
    GradientGVFN(
        initθ(length(horde), in + length(horde)), # θ
        Flux.zeros(length(horde), in + length(horde)), # h
        Flux.zeros(length(horde), in + length(horde)), # Δθ
        Flux.zeros(length(horde), in + length(horde)), # Δh
        Flux.zeros(length(horde), in + length(horde)), # Ψ
        [Flux.zeros(length(horde), in + length(horde))], # Φ FIXME: These dimensions are definitely wrong
        [Flux.zeros(length(horde), in + length(horde))], # Φ′
        [Flux.zeros((in + length(horde))*length(horde), (in + length(horde))*length(horde)) for i in 1:length(horde)], # H FIXME: These dimensions are probably wrong and need fixed to fit with the rest of the new notation.
        [Flux.zeros((in + length(horde))*length(horde), (in + length(horde))*length(horde)) for i in 1:length(horde)], # ΔH
        horde,
        σ
    )

(gvfn::GradientGVFN)(s_tp1, pred_t, type::Type{Prediction}) = gvfn.σ.(gvfn.θ*[s_tp1; pred_t], type)
(gvfn::GradientGVFN)(s_tp1, pred_t, type::Type{Derivative}) = gvfn.σ.(gvfn.θ*[s_tp1; pred_t], type)
(gvfn::GradientGVFN)(s_tp1, pred_t, type::Type{DoubleDerivative}) = gvfn.σ.(gvfn.θ*[s_tp1; pred_t], type)

# recur_roll(gvfn::GradientGVFN, state::Array{<:AbstractFloat, 1}, pred_init, type) = [gvfn(state, pred_init, type)]
# function recur_roll(gvfn::GradientGVFN, states, pred_init, type)
#     if length(states) == 1
#         return [gvfn(states[end], pred_init, type)]
#     end
#     preds = recur_roll(gvfn, states[1:(end-1)], pred_init, type)
#     push!(preds, gvfn(states[end], preds[end], type))
# end

function roll(gvfn::GradientGVFN, states, pred_init, type)
    preds = [gvfn(states[1], pred_init, type)]
    for i in 2:length(states)
        push!(preds, gvfn(states[i], preds[i-1], type))
    end
    return preds
end


is_cumulant(cumulant, j) = false
is_cumulant(cumulant::PredictionCumulant, j) = cumulant.idx == j

is_cumulant(gvfn::T, i, j) where {T <: AbstractGVFLayer} = is_cumulant(gvfn.horde[i].cumulant)

mutable struct RGTD <: LearningUpdate
    α::Float64
    β::Float64
end


function update!(gvfn::GradientGVFN{H}, lu::RGTD, h_init, states, env_state_tp1) where {H <: AbstractHorde}

    preds = roll(gvfn, states, h_init, Prediction)
    preds′ = roll(gvfn, states, h_init, Derivative)
    preds′′ = roll(gvfn, states, h_init, DoubleDerivative)

    
    
    
end

# function update!(gvfn::Flux.Recur{T}, lu::RTDC, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

#     # α = lu.α

#     reset!(gvfn, h_init)
#     preds = gvfn.(states)
#     preds_t = preds[end-1]
#     preds_tilde = Flux.data(preds[end])

#     ϕ = jacobian(preds[end-1], params(gvfn.cell))
    

#     cumulants, discounts, ρ = get(gvfn.cell, preds_tilde, env_state_tp1)
#     targets = cumulants .+ discounts.*preds_tilde
#     δ = targets .- preds_t

#     prms = params(gvfn.cell)
#     # println(prms)
#     grads = Tracker.gradient(() ->mean(δ.^2), prms)
#     for weights in prms
#         Flux.Tracker.update!(weights, -α.*grads[weights])
#     end



    
# end
