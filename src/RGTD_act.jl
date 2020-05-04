
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
function get_θ(gvfn::GradientGVFN_act, act) 
    place = ((act-1)*get_npk(gvfn))
    @view gvfn.θ[:, place+1:(place+get_npk(gvfn))]
end

(gvfn::GradientGVFN_act)(s_tp1, pred_t, type) =
    gvfn.σ.(get_θ(gvfn, s_tp1[1])*[s_tp1[2]; pred_t], type)
# (gvfn::GradientGVFN_act)(s_tp1, pred_t, type::Type{Derivative}) =
#     gvfn.σ.(get_θ(gvfn, s_tp1[1])*[s_tp1[2]; pred_t], type)
# (gvfn::GradientGVFN_act)(s_tp1, pred_t, type::Type{DoubleDerivative}) =
#     gvfn.σ.(get_θ(gvfn, s_tp1[1])*[s_tp1[2]; pred_t], type)

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

function roll(gvfn::GradientGVFN_act, states, pred_init, preds, type)
    ret = [gvfn(states[1], pred_init, type)]
    for i in 2:length(states)
        push!(ret, gvfn(states[i], preds[i-1], type))
    end
    return ret
end

function get_rolls(gvfn::GradientGVFN_act, states, h_init)
    preds = roll(gvfn, states, h_init, Prediction)
    preds′ = roll(gvfn, states, h_init, preds, Derivative)
    preds′′ = roll(gvfn, states, h_init, preds, DoubleDerivative)
    preds, preds′, preds′′
end


is_cumulant(gvfn::T, i, j) where {T <: GradientGVFN_act} =
    is_cumulant(gvfn.horde[i].cumulant, j)

function is_cumulant_mat(gvfn::T) where {T<:GradientGVFN_act}
    C = BitMatrix(undef, gvfn.n, gvfn.n)
    for (i, j) in Iterators.product(1:gvfn.n, 1:gvfn.n)
        C[j,i] = is_cumulant(gvfn, i, j)
    end
    C
end



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

    kj_iter = collect(Iterators.product(1:n, 1:num_feats))

    preds, preds′, preds′′ = get_rolls(gvfn, states, h_init)

    preds_t = preds[end-1]
    preds_tilde = preds[end]

    # calculate the gradients:
    # Assume the gradients are zero at the beginning of input (I.e. truncated).
    fill!(ϕ, zero(eltype(ϕ)))
    fill!(ϕ', zero(eltype(ϕ)))
    fill!(ξ, zero(eltype(ξ)))
    fill!(hvp, zero(eltype(hvp)))

    gradR = zeros(Float32, num_feats)
    xtp1 = zeros(Float32, num_feats)
    
    @inbounds for tp1 in 1:(length(states)-1)
        
        act_t = states[tp1][1]
        place = (act_t-1)*npk
        
        fill!(xtp1, 0.0f0)
        fill!(gradR, 0.0f0)

        preds′_tp1 = preds′[tp1]
        preds′′_tp1 = preds′′[tp1]
        
        xtp1[place+1:place+npk] = if tp1 == 1
            [states[tp1][2]; h_init] # x_tp1 = [o_t, h_{t-1}]
        else
            [states[tp1][2]; preds[tp1-1]] # x_tp1 = [o_t, h_{t-1}]
        end

        xw = w*xtp1

        for i in 1:n
            gradR[place+no+i] = ξ[i]
        end

        θgradR = θ*gradR

        term_0 = preds′′_tp1 .* (θgradR + xw)

        w_gradJK = view(w, :, (place+no+1):(place+no+n))*ϕ
        θ_gradJK = view(θ, :, (place+no+1):(place+no+n))*ϕ

        θ_hess = view(θ, :, (place+no+1):(place+no+n))*hvp

        term_2 = θ_hess + w_gradJK
        term_1 = θ_gradJK

        @inbounds @simd for kj ∈ kj_iter
            
            k=kj[1]; j=kj[2]
            grad_place = (k-1)*num_feats + j
            
            term_1[k, grad_place] += xtp1[j]
            term_2[k, grad_place] += gradR[j]
        end

        hvp .= term_0.*term_1 + preds′_tp1 .* term_2
        ϕ .= preds′_tp1 .* term_1
        ξ .= preds′_tp1 .* (θgradR + xw)
    end

    tp1 = length(states)

    act_t = states[end][1]
    place = (act_t-1)*npk
    
    fill!(xtp1, 0.0f0)
    xtp1[place+1:place+npk] = [states[end][2]; preds[end-1]] # x_tp1 = [o_t, h_{t-1}]
    preds′_tp1 = preds′[tp1]

    term_1 = θ[:, (place+no+1):(place+no+n)]*ϕ
    
    @inbounds @simd for kj ∈ kj_iter
        k = kj[1]; j = kj[2]
        grad_place = (k-1)*num_feats + j
        term_1[k, grad_place] += xtp1[j]
    end

    ϕ′ .= preds′_tp1 .* term_1
    
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

function update_old!(gvfn::GradientGVFN_act{H},
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

    preds, preds′, preds′′ = get_rolls(gvfn, states, h_init)

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
        fill!(xtp1, 0.0f0)
        fill!(gradR, 0.0f0)
        fill!(gradJK, 0.0f0)
        fill!(hess, 0.0f0)

        preds′_tp1 = preds′[tp1]
        preds′′_tp1 = preds′′[tp1]
        
        xtp1[place+1:place+npk] = if tp1 == 1
            [states[tp1][2]; h_init] # x_tp1 = [o_t, h_{t-1}]
        else
            [states[tp1][2]; preds[tp1-1]] # x_tp1 = [o_t, h_{t-1}]
        end

        xw = w*xtp1

        for i in 1:n
            gradR[place+no+i] = ξ[i]
        end

        θgradR = θ*gradR

        term_0 = preds′′_tp1 .* (θgradR + xw)
        
        for (k,j) ∈ Iterators.product(1:n, 1:num_feats)
            
            grad_place = (k-1)*num_feats + j
            # gradJK[place+no+1:place+no+n] = ϕ[:, grad_place]
            for i ∈ 1:n
                gradJK[place+no+i] = ϕ[i, grad_place]
            end

            term_1 = θ*gradJK
            term_1[k] += xtp1[j]

            for i ∈ 1:n
                hess[place+no+i] = hvp[i, grad_place]
            end

            term_2 = θ*hess + w*gradJK
            term_2[k] += gradR[j]
            
            delta_hvp[:, grad_place] =
                term_0 .* term_1 +
                preds′_tp1 .* term_2

            ϕ′[:, grad_place] = preds′_tp1 .* term_1
        end

        hvp .= delta_hvp
        ϕ .= ϕ′

        ξ .= preds′_tp1 .* (θgradR + xw)
    end

    tp1 = length(states)

    act_t = states[end][1]
    place = (act_t-1)*npk
    fill!(xtp1, 0.0f0)
    fill!(gradJK, 0.0f0)
    xtp1[place+1:place+npk] = [states[end][2]; preds[end-1]] # x_tp1 = [o_t, h_{t-1}]
    preds′_tp1 = preds′[tp1]
    for (k,j) ∈ Iterators.product(1:n, 1:num_feats)
        grad_place = (k-1)*num_feats + j
        for i ∈ 1:n
            gradJK[place+no+i] = ϕ[i, grad_place]
        end
        term_1 = θ*gradJK
        term_1[k] += xtp1[j]
        ϕ′[:, grad_place] = preds′_tp1 .* term_1
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


