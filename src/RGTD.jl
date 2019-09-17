

struct Prediction end
struct Derivative end
struct DoubleDerivative end

sigmoid(x, type::Type{Prediction}) = sigmoid(x)
sigmoid(x, type::Type{Derivative}) = sigmoid′(x)
sigmoid(x, type::Type{DoubleDerivative}) = sigmoid′′(x)

relu(x, type::Type{Prediction}) = Flux.relu(x)
relu(x, type::Type{Derivative}) = x > 0 ? one(x) : zero(x)
relu(x, type::Type{DoubleDerivative}) = 0

linear(x, type::Type{Prediction}) = identity(x)
linear(x, type::Type{Derivative}) = one(x) 
linear(x, type::Type{DoubleDerivative}) = 0



"""
    GradientGVFN

A GVFN only for implementing recurrent GTD, which can be used with multiple activation functions (sigmoid, relu, linear). We will use the following as short hand in all the following notes:
    - n ∈ Z : The number of GVFs in the GVFN
    - k ∈ Z : The number of inputs in the observations

Parameters:
    θ::Matrix{Float32} -> The weights of the function approximator to make the next hidden state (size: n×(n+k)
    h::Matrix{Float32} -> The secondary weights for the GTD (TDC) update.
    
    Δθ::Matrix{Float32} -> The change in weights (or the update matrix)
    Δh::Matrix{Float32} -> The change in secondary weights (or the secondary update matrix)
    
    # Psi
    Ψ::Matrix{Float32} -> For storing the Hessian operations
    # Phi
    Φ::Matrix{Float32} -> For storing the gradients calculated through time (n × (n*(n+k)))
    # PhiPrime
    Φ′::Matrix{Float32} -> For storing the gradients calculated through time for the s_{t+1}
    # H
    H::Vector{Matrix{Float32}} -> The Hessian Matrix (This will store the hessian vector product)
    # deltaH
    ΔH::Vector{Matrix{Float32}}
"""
mutable struct GradientGVFN{H<:AbstractHorde, F<:Function}
    
    # Weights
    θ::Matrix{Float32}
    # Secondary Weights
    h::Matrix{Float32}

    # Psi
    Ψ::Vector{Float32}

    # eta A helper variable for calculating efficiently n \by n(n+k)
    η::Matrix{Float32}
    ξ::Matrix{Float32}
    # Phi
    # Number of GVFs \times Number of weights (n*(n+length(s)))
    Φ::Matrix{Float32}
    # PhiPrime
    Φ′::Matrix{Float32}
    # H
    Hvp::Matrix{Float32}
    # deltaH

    # Questions Graph (𝒳)
    # GVFs
    horde::H

    # Activation
    σ::F
    # number of observations:
    k::Int64
    # number of GVFs
    n::Int64
end

function GradientGVFN(in, horde, σ; initθ=Flux.glorot_uniform)
    n = length(horde)
    npk = in + length(horde)
    GradientGVFN(
        initθ(n, npk), # θ
        zeros(Float32, n, npk), # h
        zeros(Float32, n*npk), # Ψ
        zeros(Float32, n, n*npk), # η
        zeros(Float32, n), # ξ
        zeros(Float32, n, n*npk), # Φ 
        zeros(Float32, n, n*npk), # Φ′
        zeros(Float32, n, n*npk), # H
        horde,
        σ,
        in,
        n)
end

(gvfn::GradientGVFN)(s_tp1, pred_t, type::Type{Prediction}) = gvfn.σ.(gvfn.θ*[s_tp1; pred_t], type)
(gvfn::GradientGVFN)(s_tp1, pred_t, type::Type{Derivative}) = gvfn.σ.(gvfn.θ*[s_tp1; pred_t], type)
(gvfn::GradientGVFN)(s_tp1, pred_t, type::Type{DoubleDerivative}) = gvfn.σ.(gvfn.θ*[s_tp1; pred_t], type)

"""
    roll

This takes a gvfn, states, initial state, and a type and produces a list of preds with a function
"""
function roll(gvfn::GradientGVFN, states, pred_init, type)
    preds = [gvfn(states[1], pred_init, type)]
    for i in 2:length(states)
        push!(preds, gvfn(states[i], preds[i-1], type))
    end
    return preds
end


is_cumulant(cumulant, j) = false
is_cumulant(cumulant::PredictionCumulant, j) =
    cumulant.idx == j

is_cumulant(gvfn::T, i, j) where {T <: AbstractGVFLayer} =
    is_cumulant(gvfn.horde[i].cumulant)

mutable struct RGTD <: LearningUpdate
    α::Float64
    β::Float64
end

"""
    _sum_kron_delta(lhs::matrix, rhs::vector)

    lhs = i \by npk*n, rhs = npk

lhs^i_{k,j} + rhs_j *δ^κ_{i,k}
"""
function _sum_kron_delta!(lhs, rhs, npk)
    for i in 1:size(lhs)[2]
        idx = npk*(i-1)+1
        lhs[i, idx:(idx+npk)] .+= rhs
    end
end

function update!(gvfn::GradientGVFN{H},
                 opt,
                 lu::RGTD,
                 h_init,
                 states,
                 env_state_tp1,
                 action_t=nothing,
                 b_prob::Float32=1.0f0) where {H <: AbstractHorde}

    η = gvfn.η
    ϕ = gvfn.ϕ
    ϕ′ = gvfn.ϕ′
    θ = gvfn.θ
    w = gvfn.w
    
    preds = roll(gvfn, states, h_init, Prediction)
    preds′ = roll(gvfn, states, h_init, Derivative)
    preds′′ = roll(gvfn, states, h_init, DoubleDerivative)

    preds_t = preds[end-1]
    preds_tilde = preds[end]

    # calculate the gradients:
    # Assume the gradients are zero at the beginning of input (I.e. truncated).
    fill!(ϕ, zero(eltype(ϕ)))
    fill!(ϕ', zero(eltype(ϕ)))


    xtp1 = [states[1]; h_init] # x_tp1 = [o_t, h_{t-1}]
    # xtp1 = [states[2]; preds[1]] # x_tp1 = [o_tp1, h_t]
    # Time Step t
    # η n\byn*(n+k)
    η .= _sum_kron_delta(θ[:, k:end]*ϕ, xtp1) # tp1
    ϕ′ .= preds′[1] .* η # tp1
    # ut_{k,j} = ϕ(t)_{k,j}
    wx = w*xtp1
    # ξ is n+k
    term_1 = (preds′′[1].*(θ[:, k:end]*ξ .+ wx)) .* η
    term_2 = preds′[1] .* _sum_kron_delta(θ[:, k:end]*H .+ w[:, k:end]*ϕ, [zero(states[1]); ξ])

    H .= term_1 .+ term_2

    ξ .= preds′*(θ[:, k:end]*ξ + wx)

    ϕ .= ϕ′ # tp1 -> t
    for tp1 in 2:length(states)
        xtp1 = [states[tp1]; preds[tp1-1]] # x_tp1 = [o_t, h_{t-1}]
        η .= _sum_kron_delta!(θ[:, k:end]*ϕ, xtp1)
        ϕ′ .= preds′[tp1] .* η

        wx = w*xtp1
        # ξ is n+k
        term_1 = (preds′′[1].*(θ[:, k:end]*ξ .+ wx)) .* η
        term_2 = preds′[1] .* _sum_kron_delta(θ[:, k:end]*H .+ w[:, k:end]*ϕ, [zero(states[1]); ξ])

        H .= term_1 .+ term_2

        ξ .= preds′*(θ[:, k:end]*ξ + wx)
        ϕ .= ϕ′ # tp1 -> t
    end

    cumulants, discounts, π_prob = get(gvfn.cell, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob ./ b_prob
    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    ϕw = ϕ*reshape(w', length(w), 1)

    rs_θ = reshape(θ, length(θ), 1)

    ψ .= sum((ρ.*δ .- ϕw).*H; dims=1)
    
    rs_θ .+= α.*(ρ.*δ).*ϕ .- (ρ.*ϕw) .* (C*ϕ′ .+ discounts.*ϕ′) .- ψ
    
    rs_w = reshape(w', length(w), 1)
    rs_w .+= β.*sum((ρ.*δ).*ϕ - (ϕw .* ϕ); dims=1)
end


