

struct Prediction end
struct Derivative end
struct DoubleDerivative end



sigmoid(x, type::Type{Prediction}) = sigmoid(x)
sigmoid(x, type::Type{Derivative}) = sigmoid′(x)
sigmoid(x, type::Type{DoubleDerivative}) = sigmoid′′(x)

σ(x) = sigmoid(x)
σ(x, type::Type{Prediction}) = sigmoid(x)
σ(x, type::Type{Derivative}) = sigmoid′(x)
σ(x, type::Type{DoubleDerivative}) = sigmoid′′(x)

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
        zeros(Float32, n, n*npk, n*npk), # Hessian, i, kj, mn
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
function roll(gvfn::GradientGVFN, states, pred_init, type::Type{Prediction})
    preds = [gvfn(states[1], pred_init, type)]
    for i in 2:length(states)
        push!(preds, gvfn(states[i], preds[i-1], type))
    end
    return preds
end

function roll(gvfn::GradientGVFN, states, pred_init, type)
    preds = roll(gvfn, states, pred_init, Prediction)
    ret = [gvfn(states[1], pred_init, type)]
    for i in 2:length(states)
        push!(ret, gvfn(states[i], preds[i-1], type))
    end
    return ret
end


is_cumulant(cumulant, j) = false
is_cumulant(cumulant::PredictionCumulant, j) =
    cumulant.idx == j

is_cumulant(gvfn::T, i, j) where {T <: GradientGVFN} =
    is_cumulant(gvfn.horde[i].cumulant, j)

function is_cumulant_mat(gvfn::T) where {T<:GradientGVFN}
    C = BitMatrix(undef, gvfn.n, gvfn.n)
    for (i, j) in Iterators.product(1:gvfn.n, 1:gvfn.n)
        C[j,i] = is_cumulant(gvfn, i, j)
    end
    C
end

mutable struct RGTD <: LearningUpdate
    α::Float32
    β::Float32
    term_1::IdDict
    term_2::IdDict
    RGTD(α, β) = new(α, β, IdDict(), IdDict())
end

"""
    _sum_kron_delta(lhs::matrix, rhs::vector)

    lhs = i \by npk*n, rhs = npk

lhs^i_{k,j} + rhs_j *δ^κ_{i,k}
"""
function _sum_kron_delta!(lhs, rhs, k, n)
    for i in 1:size(lhs)[1]
        idx = (n+k)*(i-1)+1
        lhs[i, idx:(idx+(n+k-1))] .+= rhs
    end
    return lhs
end

function _sum_kron_delta(lhs, rhs, k, n)
    ret = copy(lhs)
    for i in 1:size(lhs)[1]
        idx = (n+k)*(i-1)+1
        ret[i, idx:(idx+(n+k-1))] .+= rhs
    end
    ret
end

function _sum_kron_delta_2!(lhs, k, val)
    lhs[k] += val
    lhs
end

_kron_delta(x::T, i, j) where {T<:Number} = i==j ? one(x) : zero(x)





function update!(gvfn::GradientGVFN{H},
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
    k = gvfn.k
    n = gvfn.n
    ξ = gvfn.ξ
    hvp = gvfn.Hvp
    
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

    for tp1 in 1:(length(states)-1)
        xtp1 = if tp1 == 1
            [states[tp1]; h_init] # x_tp1 = [o_t, h_{t-1}]
        else
            [states[tp1]; preds[tp1-1]] # x_tp1 = [o_t, h_{t-1}]
        end
        η .= _sum_kron_delta(θ[:, (k+1):end]*ϕ, xtp1, k, n)
        ϕ′ .= preds′[tp1] .* η
        wx = w*xtp1

        # ξ is n+k
        term_1 = (preds′′[tp1].*(θ[:, (k+1):end]*ξ .+ wx)) .* η
        term_2 = preds′[tp1] .* _sum_kron_delta(θ[:, (k+1):end]*hvp .+ w[:, (k+1):end]*ϕ, [zero(states[1]); ξ], k, n)
        hvp .= term_1 .+ term_2
        ξ .= preds′[tp1].*(θ[:, (k+1):end]*ξ + wx)
        ϕ .= ϕ′ # tp1 -> t
    end
    
    xtp1 = [states[end]; preds[end-1]] # x_tp1 = [o_t, h_{t-1}]
    η .= _sum_kron_delta(θ[:, (k+1):end]*ϕ, xtp1, k, n)
    ϕ′ .= preds′[end] .* η
    
    cumulants, discounts, π_prob = get(gvfn.horde, action_t, env_state_tp1, preds_tilde)
    ρ = π_prob ./ b_prob
    targets = cumulants + discounts.*preds_tilde
    δ =  targets - preds_t

    ϕw = ϕ*reshape(w', length(w), 1)

    rs_θ = reshape(θ', length(θ),)
    Ψ .= sum((ρ.*δ - ϕw).*hvp; dims=1)[1,:]
    α = lu.α
    β = lu.β
    C = is_cumulant_mat(gvfn)
    
    rs_θ .+= α*(sum(((ρ.*δ).*ϕ - (ρ.*ϕw) .* (C'*ϕ′ .+ discounts.*ϕ′)); dims=1)[1,:] - Ψ)

    rs_w = reshape(w', length(w),)
    rs_w .+= β*sum(((ρ.*δ - ϕw) .* ϕ); dims=1)[1,:]
end

function update_full_hessian!(gvfn::GradientGVFN{H},
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
    num_feats = gvfn.k
    num_gvfs = gvfn.n
    ξ = gvfn.ξ
    hvp = gvfn.Hvp
    hess = gvfn.Hessian

    nd = num_gvfs+num_feats
    
    _hess = zeros(nd)
    _gradMN = zeros(nd)
    _gradJK = zeros(nd)
    new_H = zero(hess)

    _kd_mat = Diagonal(ones(Int64, num_gvfs))
    
    preds = roll(gvfn, states, h_init, Prediction)
    preds′ = roll(gvfn, states, h_init, Derivative)
    preds′′ = roll(gvfn, states, h_init, DoubleDerivative)

    # calculate the gradients:
    # Assume the gradients are zero at the beginning of input (I.e. truncated).
    fill!(ϕ, zero(eltype(ϕ)))
    # fill!(ϕ', zero(eltype(ϕ)))
    fill!(ξ, zero(eltype(ξ)))
    # fill!(Ψ, zero(eltype(Ψ)))
    # fill!(hess, zero(eltype(hess)))
    # fill!(hvp, zero(eltype(hvp)))

    @inbounds for t in 1:(length(states)-1)

        x_t = if t == 1
            [states[t]; h_init]
        else
            [states[t]; preds[t-1]] # x_tp1 = [o_t, h_{t-1}]
        end


        
        # println(x_t)

        nd = num_feats+num_gvfs
            
        @inbounds for (k,j) in Iterators.product(1:num_gvfs, 1:(nd))

            @inbounds _gradJK[(num_feats+1):end] .= ϕ[:, (k-1)*(nd) + j]
                
            @inbounds for (m,n) in Iterators.product(1:num_gvfs, 1:(nd))

                @inbounds _hess[(num_feats+1):end] .= hess[:, (k-1) * nd + j, (m-1) * (nd) + n]
                @inbounds _gradMN[(num_feats+1):end] .= ϕ[:, (m-1)*nd + n];

                @inbounds for i in 1:num_gvfs

                    @inbounds new_H[i, (k-1) * nd + j, (m-1) * nd + n] =
                        preds′′[t][i] * (dot(_gradMN, θ[i, :]) + x_t[n]*_kd_mat[i, m]) * (dot(_gradJK, θ[i,:]) + x_t[j]*_kd_mat[i,k]) +
                        preds′[t][i] * (dot(_hess, θ[i, :]) + _gradJK[n]*_kd_mat[i, m] + _hess[j]*_kd_mat[i,k])

                end
            end
        end

        hess .= new_H

        @inbounds for (k,j) in Iterators.product(1:num_gvfs, 1:(nd))
            # ϕ[:, (k-1)*(nd) + j] = preds′[tp1] .* (_sum_kron_delta_2!(θ[:, (num_feats+1):end]*ϕ[:, (k-1)*(nd) + j], k, xtp1[j]))
            _gradJK[(num_feats+1):end] .= ϕ[:, (k-1)*(nd) + j]
            ϕ[:, (k-1)*(nd) + j] .= preds′[t] .* (_sum_kron_delta_2!(θ*_gradJK, k, x_t[j]))
        end
        
    end

    xtp1 = [states[end]; preds[end-1]] # x_tp1 = [o_tp1, h_{t}]
    # η .= _sum_kron_delta(θ[:, (num_feats+1):end]*ϕ, xtp1, num_feats, num_gvfs)
    # ϕ′ .= preds′[end] .* η

    @inbounds for (k,j) in Iterators.product(1:num_gvfs, 1:(nd))
        _gradJK[(num_feats+1):end] .= ϕ[:, (k-1)*(nd) + j]
        ϕ′[:, (k-1)*(nd) + j] = preds′[end] .* _sum_kron_delta_2!(θ*_gradJK, k, xtp1[j])
    end

    cumulants, discounts, π_prob = get(gvfn.horde, action_t, env_state_tp1, preds[end])
    ρ = π_prob ./ b_prob
    targets = cumulants + discounts.*preds[end]
    δ =  targets .- preds[end-1]
    # println(δ)


    ϕw = ϕ*reshape(w', length(w), 1)

    rs_θ = reshape(θ', length(θ), 1)

    # @inbounds for i in 1:num_gvfs
    #     hvp[i,:] .= hess[i,:,:]*reshape(w', length(w), )
    # end
    
    # Ψ .= sum([(ρ[i]*δ[i] - ϕw[i]).*(hess[i,:,:]*reshape(w', length(w), )) for i in 1:num_gvfs])
    for i in 1:num_gvfs
        Ψ .+= (ρ[i]*δ[i] - ϕw[i])*(hess[i,:,:]*reshape(w', length(w), ))
    end

    α = lu.α
    β = lu.β
    C = is_cumulant_mat(gvfn)

    rs_θ .+= α.*(ϕ'*(ρ.*δ) - ((C'*ϕ′)' * (ρ.*ϕw)) .- Ψ)


    
    rs_w = reshape(w', length(w), 1)
    rs_w .+= β.*(ϕ'*(ρ.*δ - ϕw))

end

function update_full_hessian_fast!(gvfn::GradientGVFN{H},
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
    num_feats = gvfn.k
    num_gvfs = gvfn.n
    ξ = gvfn.ξ
    hvp = gvfn.Hvp
    hess = gvfn.Hessian

    nd = num_gvfs+num_feats
    
    _hess = zeros(nd)
    _gradMN = zeros(nd)
    _gradJK = zeros(nd)
    new_H = zero(hess)

    _kd_mat = diagm(ones(Int64, num_gvfs))
    
    preds = roll(gvfn, states, h_init, Prediction)
    preds′ = roll(gvfn, states, h_init, Derivative)
    preds′′ = roll(gvfn, states, h_init, DoubleDerivative)

    # calculate the gradients:
    # Assume the gradients are zero at the beginning of input (I.e. truncated).
    fill!(ϕ, zero(eltype(ϕ)))
    fill!(ξ, zero(eltype(ξ)))
    fill!(Ψ, zero(eltype(Ψ)))

    v_gradJK = @view _gradJK[(num_feats+1):end]
    v_gradMN = @view _gradMN[(num_feats+1):end]
    v_hess = @view _hess[(num_feats+1):end]

    @inbounds for t in 1:(length(states)-1)

        x_t = if t == 1
            [states[t]; h_init]
        else
            [states[t]; preds[t-1]] # x_tp1 = [o_t, h_{t-1}]
        end


        nd = num_feats+num_gvfs
            
        @inbounds for (k,j) in Iterators.product(1:num_gvfs, 1:(nd))

            v_gradJK .= ϕ[:, (k-1)*nd + j]
            θ_gradJK = θ*_gradJK

            @inbounds for (m,n) in Iterators.product(1:num_gvfs, 1:(nd))

                for i in 1:num_gvfs
                    v_hess[i] = hess[i, (k-1) * nd + j, (m-1) * (nd) + n]
                    v_gradMN[i] = ϕ[i, (m-1)*nd + n]
                end

                θ_gradMN = θ*_gradMN
                θ_hess = θ*_hess
                
                @inbounds new_H[:, (k-1) * nd + j, (m-1) * nd + n] =
                    preds′′[t] .* (θ_gradMN .+ x_t[n]*_kd_mat[m, :]) .* (θ_gradJK .+ x_t[j]*_kd_mat[k, :]) +
                    preds′[t] .* (θ_hess .+ _gradJK[n]*_kd_mat[m, :] .+ _hess[j]*_kd_mat[k, :])


            end
        end

        hess .= new_H

        @inbounds for (k,j) in Iterators.product(1:num_gvfs, 1:(nd))
            # ϕ[:, (k-1)*(nd) + j] = preds′[tp1] .* (_sum_kron_delta_2!(θ[:, (num_feats+1):end]*ϕ[:, (k-1)*(nd) + j], k, xtp1[j]))
            _gradJK[(num_feats+1):end] .= ϕ[:, (k-1)*(nd) + j]
            ϕ[:, (k-1)*(nd) + j] .= preds′[t] .* (_sum_kron_delta_2!(θ*_gradJK, k, x_t[j]))
        end
        
    end

    xtp1 = [states[end]; preds[end-1]] # x_tp1 = [o_tp1, h_{t}]
    # η .= _sum_kron_delta(θ[:, (num_feats+1):end]*ϕ, xtp1, num_feats, num_gvfs)
    # ϕ′ .= preds′[end] .* η

    @inbounds for (k,j) in Iterators.product(1:num_gvfs, 1:(nd))
        _gradJK[(num_feats+1):end] .= ϕ[:, (k-1)*(nd) + j]
        ϕ′[:, (k-1)*(nd) + j] = preds′[end] .* _sum_kron_delta_2!(θ*_gradJK, k, xtp1[j])
    end

    cumulants, discounts, π_prob = get(gvfn.horde, action_t, env_state_tp1, preds[end])
    ρ = π_prob ./ b_prob
    targets = cumulants + discounts.*preds[end]
    δ =  targets .- preds[end-1]


    ϕw = ϕ*reshape(w', length(w), 1)

    rs_θ = reshape(θ', length(θ), 1)

    for i in 1:num_gvfs
        Ψ .+= (ρ[i]*δ[i] - ϕw[i])*(hess[i,:,:]*reshape(w', length(w), ))
    end

    α = lu.α
    β = lu.β
    C = is_cumulant_mat(gvfn)

    rs_θ .+= α.*(ϕ'*(ρ.*δ) - ((C'*ϕ′)' * (ρ.*ϕw)) .- Ψ)


    
    rs_w = reshape(w', length(w), 1)
    rs_w .+= β.*(ϕ'*(ρ.*δ - ϕw))

end


function update!(model::SingleLayer, horde::AbstractHorde, opt, lu::RGTD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)

    v = model.(state_seq[end-1:end])
    v_prime_t = deriv(model, state_seq[end-1])

    c, γ, π_prob = get(horde, action_t, env_state_tp1, Flux.data(v[end]))
    ρ = π_prob./b_prob
    δ = ρ.*tderror(v[end-1], c, γ, Flux.data(v[end]))
    Δ = δ.*v_prime_t
    model.W .-= apply!(opt, model.W, Δ*state_seq[end-1]')
    model.b .-= apply!(opt, model.b, Δ)
end

