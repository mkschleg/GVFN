
# struct Prediction end
# struct Derivative end
# struct DoubleDerivative end



# sigmoid(x, type::Type{Prediction}) = sigmoid(x)
# sigmoid(x, type::Type{Derivative}) = sigmoid′(x)
# sigmoid(x, type::Type{DoubleDerivative}) = sigmoid′′(x)

# σ(x) = sigmoid(x)
# σ(x, type::Type{Prediction}) = sigmoid(x)
# σ(x, type::Type{Derivative}) = sigmoid′(x)
# σ(x, type::Type{DoubleDerivative}) = sigmoid′′(x)

# relu(x, type::Type{Prediction}) = Flux.relu(x)
# relu(x, type::Type{Derivative}) = x > 0 ? one(x) : zero(x)
# relu(x, type::Type{DoubleDerivative}) = 0

# linear(x, type::Type{Prediction}) = identity(x)
# linear(x, type::Type{Derivative}) = one(x) 
# linear(x, type::Type{DoubleDerivative}) = 0


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
mutable struct GradientGVFN_act{H<:AbstractHorde, F<:Function}
    
    # Weights
    θ::Matrix{Float32}
    # Secondary Weights
    h::Matrix{Float32}

    # Psi
    Ψ::Vector{Float32}

    # eta A helper variable for calculating efficiently n \by n(n+k)
    η::Matrix{Float32}
    ξ::Vector{Float32}
    # Phi
    # Number of GVFs \times Number of weights (n*(n+length(s)))
    Φ::Matrix{Float32}
    # PhiPrime
    Φ′::Matrix{Float32}
    # H
    Hvp::Matrix{Float32}
    Hessian::Array{Float32, 3}
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
    # number of actions
    a::Int64
end

function GradientGVFN_act(in, horde, num_actions, σ; initθ=Flux.glorot_uniform)
    n = length(horde)
    npk = in + length(horde)
    GradientGVFN_act(
        initθ(n, npk*num_actions), # θ
        zeros(Float32, n, npk*num_actions), # h
        zeros(Float32, n*npk*num_actions), # Ψ
        zeros(Float32, n, n*npk*num_actions), # η
        zeros(Float32, n), # ξ
        zeros(Float32, n, n*npk*num_actions), # Φ 
        zeros(Float32, n, n*npk*num_actions), # Φ′
        zeros(Float32, n, n*npk*num_actions), # H
        zeros(Float32, n, n*npk*num_actions, n*npk*num_actions), # Hessian, i, kj, mn
        horde,
        σ,
        in,
        n,
        num_actions)
end

get_npk(gvfn::GradientGVFN_act) = gvfn.n + gvfn.k
get_θ(gvfn::GradientGVFN_act, act) = begin
    place = ((act-1)*get_npk(gvfn))
    # @show size(gvfn.θ[:, place:(place-1+get_npk(gvfn))])
    @view gvfn.θ[:, place+1:(place+get_npk(gvfn))]
end

(gvfn::GradientGVFN_act)(s_tp1, pred_t, type::Type{Prediction}) =
    gvfn.σ.(get_θ(gvfn, s_tp1[1])*[s_tp1[2]; pred_t], type)
(gvfn::GradientGVFN_act)(s_tp1, pred_t, type::Type{Derivative}) =
    gvfn.σ.(get_θ(gvfn, s_tp1[1])*[s_tp1[2]; pred_t], type)
(gvfn::GradientGVFN_act)(s_tp1, pred_t, type::Type{DoubleDerivative}) =
    gvfn.σ.(get_θ(gvfn, s_tp1[1])*[s_tp1[2]; pred_t], type)

"""
    roll

This takes a gvfn, states, initial state, and a type and produces a list of preds with a function
"""
function roll(gvfn::GradientGVFN_act, states, pred_init, type::Type{Prediction})
    preds = [gvfn(states[1], pred_init, type)]
    for i in 2:length(states)
        push!(preds, gvfn(states[i], preds[i-1], type))
    end
    return preds
end

function roll(gvfn::GradientGVFN_act, states, pred_init, type)
    preds = roll(gvfn, states, pred_init, Prediction)
    ret = [gvfn(states[1], pred_init, type)]
    for i in 2:length(states)
        push!(ret, gvfn(states[i], preds[i-1], type))
    end
    return ret
end


# is_cumulant(cumulant, j) = false
# is_cumulant(cumulant::PredictionCumulant, j) =
#     cumulant.idx == j

is_cumulant(gvfn::T, i, j) where {T <: GradientGVFN_act} =
    is_cumulant(gvfn.horde[i].cumulant, j)

function is_cumulant_mat(gvfn::T) where {T<:GradientGVFN_act}
    C = BitMatrix(undef, gvfn.n, gvfn.n)
    for (i, j) in Iterators.product(1:gvfn.n, 1:gvfn.n)
        C[j,i] = is_cumulant(gvfn, i, j)
    end
    C
end

# abstract type AbstractGradUpdate end

# mutable struct RGTD <: AbstractGradUpdate
#     α::Float32
#     β::Float32
# end

"""
    _sum_kron_delta(lhs::matrix, rhs::vector)

    lhs = i \by npk*n, rhs = npk

lhs^i_{k,j} + rhs_j *δ^κ_{i,k}
"""
# function _sum_kron_delta!(lhs, rhs, k, n)
#     for i in 1:size(lhs)[1]
#         idx = (n+k)*(i-1)+1
#         lhs[i, idx:(idx+(n+k-1))] .+= rhs
#     end
#     return lhs
# end

# function _sum_kron_delta(lhs, rhs, k, n)
#     ret = copy(lhs)
#     for i in 1:size(lhs)[1]
#         idx = (n+k)*(i-1)+1
#         ret[i, idx:(idx+(n+k-1))] .+= rhs
#     end
#     ret
# end

# function _sum_kron_delta_2!(lhs, k, val)
#     lhs[k] += val
#     lhs
# end

# _kron_delta(x::T, i, j) where {T<:Number} = i==j ? one(x) : zero(x)

function update!(gvfn::GradientGVFN_act{H},
                 opt,
                 lu::RGTD,
                 h_init,
                 states,
                 env_state_tp1,
                 action_t=nothing,
                 b_prob::F=1.0f0) where {H <: AbstractHorde, F<:AbstractFloat}
    η = gvfn.η
    ϕ = gvfn.Φ
    ϕ′ = gvfn.Φ′
    Ψ = gvfn.Ψ
    θ = gvfn.θ
    w = gvfn.h
    no = gvfn.k
    n = gvfn.n
    ξ = gvfn.ξ
    hvp = gvfn.Hvp

    npk = n+no
    num_feats = npk*gvfn.a
    
    preds = roll(gvfn, states, h_init, Prediction)
    preds′ = roll(gvfn, states, h_init, Derivative)
    preds′′ = roll(gvfn, states, h_init, DoubleDerivative)

    preds_t = preds[end-1]
    preds_tilde = preds[end]

    # calculate the gradients:
    # Assume the gradients are zero at the beginning of input (I.e. truncated).
    fill!(ϕ, zero(eltype(ϕ)))
    fill!(ϕ', zero(eltype(ϕ)))
    fill!(ξ, zero(eltype(ξ)))
    fill!(hvp, zero(eltype(hvp)))

    delta_hvp = zero(hvp)
    
    hess = zeros(Float32, num_feats)
    gradJK = zeros(Float32, num_feats)
    gradR = zeros(Float32, num_feats)
    xtp1 = zeros(Float32, num_feats)
    for tp1 in 1:(length(states)-1)
        act_t = states[tp1][1]
        place = (act_t-1)*npk

        xtp1[place+1:place+npk] = if tp1 == 1
            [states[tp1][2]; h_init] # x_tp1 = [o_t, h_{t-1}]
        else
            [states[tp1][2]; preds[tp1-1]] # x_tp1 = [o_t, h_{t-1}]
        end

        xw = w*xtp1

        gradR[place+1+no:place+npk] = ξ
        
        for (k,j) ∈ Iterators.product(1:n, 1:num_feats)
            grad_place = (k-1)*num_feats + j
            gradJK[place+1+no:place+npk] = ϕ[:, grad_place]

            term_1 = θ*gradJK
            term_1[k] += xtp1[j]

            term_2 = θ*hess + w*gradJK
            term_2[k] += gradR[j]

            for i ∈ 1:n
                hess[place+no+i] = hvp[i, grad_place]
            end
            
            delta_hvp[:, grad_place] =
                preds′′[tp1] .* (θ*gradR + xw) .* term_1 +
                preds′[tp1] .* term_2

            ϕ′[:, grad_place] = preds′[tp1] .* term_1
        end

        hvp .= delta_hvp
        ϕ .= ϕ′

        ξ .= preds′[tp1] .* (θ*gradR + xw)
        # for i ∈ 1:n
        #     ξ[i] = preds′[tp1][i]*(dot(θ[i, :], gradR) + xw[i])
        # end

        
        
        # act_t = states[tp1][1]
        # place = (act_t-1)*npk + 1
        # xtp1 = if tp1 == 1
        #     [states[tp1][2]; h_init] # x_tp1 = [o_t, h_{t-1}]
        # else
        #     [states[tp1][2]; preds[tp1-1]] # x_tp1 = [o_t, h_{t-1}]
        # end
        
        # η .= _sum_kron_delta!(θ[:, (place+k):(place+k+n)]*ϕ, xtp1, k, n)

        # ϕ′ .= preds′[tp1] .* η
        # wx = w*xtp1

        # # ξ is n+k
        # term_1 = (preds′′[tp1].*(θ[:, (k+1):end]*ξ .+ wx)) .* η
        # term_2 = preds′[tp1] .* _sum_kron_delta!(θ[:, (k+1):end]*hvp .+ w[:, (k+1):end]*ϕ, [zero(states[1]); ξ], k, n)
        # hvp .= term_1 .+ term_2
        # ξ .= preds′[tp1].*(θ[:, (k+1):end]*ξ + wx)
        # ϕ .= ϕ′ # tp1 -> t
    end

    tp1 = length(states)

    act_t = states[end][1]
    place = (act_t-1)*npk
    xtp1[place+1:place+npk] = [states[end][2]; preds[end-1]] # x_tp1 = [o_t, h_{t-1}]
    for (k,j) ∈ Iterators.product(1:n, 1:num_feats)
        grad_place = (k-1)*num_feats + j
        gradJK[place+1+no:place+npk] = ϕ[:, grad_place]
        term_1 = θ*gradJK
        term_1[k] += xtp1[j]
        ϕ′[:, grad_place] = preds′[tp1] .* term_1
    end
    
    cumulants, discounts, π_prob = get(gvfn.horde, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob ./ b_prob
    targets = cumulants + discounts.*preds_tilde
    δ =  targets - preds_t

    ϕw = ϕ*reshape(w', length(w), 1)

    Ψ .= sum((ρ.*δ - ϕw).*hvp; dims=1)[1,:]
    α = lu.α
    β = lu.β
    C = is_cumulant_mat(gvfn)
    
    rs_θ = reshape(θ', length(θ), 1)
    rs_w = reshape(w', length(w), 1)

    rs_θ .+= α*(ϕ'*(ρ.*δ) - ((C'*ϕ′ .+ discounts.*ϕ′)' * (ρ.*ϕw)) .- Ψ)
    rs_w .+= β*(ϕ'*(ρ.*δ - ϕw))
end


