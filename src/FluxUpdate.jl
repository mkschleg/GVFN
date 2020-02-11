
using Flux


# Already defined elsewhere.
# struct TD end
#

struct TDInterpolate <: LearningUpdate
    β::Float32
end

function _gvfn_loss!(chain,
                     lu::Union{TD, TDInterpolate},
                     h_init,
                     states,
                     env_state_tp1,
                     action_t=nothing,
                     b_prob=1.0f0) where {H<:AbstractHorde}

    reset!(chain, h_init)
    preds = chain.(states)#::Array{<:TrackedArray, 1}#Array{typeof(Flux.hidden(chain[end].cell)), 1}
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, π_prob = get(chain[end].cell,
                                       action_t,
                                       env_state_tp1,
                                       preds_tilde)
    ρ = π_prob/b_prob
    return offpolicy_tdloss_gvfn(ρ,
                                 preds_t,
                                 cumulants,
                                 discounts,
                                 preds_tilde)::Tracker.TrackedReal{Float32}, preds

end

# For General Chains. 
function update!(chain,
                  horde::H,
                  opt,
                  lu::TD,
                  h_init,
                  state_seq,
                  env_state_tp1,
                  action_t=nothing,
                  b_prob=1.0f0) where {H}

    
    # Update GVFN First
    ℒ_gvfn, preds = begin
        if contains_gvfn(chain)
            gvfn_idx = find_layers_with_eq(chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)
            if length(gvfn_idx) != 1
                throw("Multi-layer GVFN Not available")
            end
            ℒ, v = _gvfn_loss!(chain[1:gvfn_idx[1]],
                               lu,
                               h_init,
                               state_seq,
                               env_state_tp1,
                               action_t,
                               b_prob;)
            # println(v[end-1])
            ℒ, chain[gvfn_idx[1]+1:end].(v[end-1:end])
        else
            reset!(chain, h_init)
            param(0.0f0), chain.(state_seq)
        end
    end

    cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = π_prob./b_prob
    ℒ_out = offpolicy_tdloss(ρ, preds[end-1], cumulants, discounts, Flux.data(preds[end]))

    grads = Flux.Tracker.gradient(()->ℒ_out + ℒ_gvfn, Flux.params(chain))
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end





function update!(chain,
                 horde::H,
                 opt,
                 lu::TDInterpolate,
                 h_init,
                 state_seq,
                 env_state_tp1,
                 action_t=nothing,
                 b_prob=1.0;
                 kwargs...) where {H}

    
    # Update GVFN First
    ℒ_gvfn = begin
        if contains_gvfn(chain)
            gvfn_idx = find_layers_with_eq(chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)
            if length(gvfn_idx) != 1
                throw("Multi-layer GVFN Not available")
            end
            _gvfn_loss!(chain[1:gvfn_idx[1]],
                        lu,
                        h_init,
                        state_seq,
                        env_state_tp1,
                        action_t,
                        b_prob;
                        kwargs...)
        else
            param(0)
        end
    end
    
    reset!(chain, h_init)
    preds = chain.(state_seq)
    cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    ℒ_out = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Flux.data(preds[end]))

    grads = Flux.Tracker.gradient(()->β*ℒ_out + (1-β)*ℒ_gvfn, Flux.params(chain))
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end


struct BatchTD <: LearningUpdate end

function _gvfn_loss!(chain,
                     lu::BatchTD,
                     h_init,
                     states,
                     env_state_tp1,
                     action_t=nothing,
                     b_prob=1.0;
                     kwargs...) where {H<:AbstractHorde}

    reset!(chain, h_init)
    preds = chain.(states)

    δ_all = param(zeros(length(preds)-1))
    for t ∈ 1:(length(preds)-1)
        preds_t = preds[t]
        preds_tilde = Flux.data(preds[t+1])

        cumulants, discounts, π_prob = get(chain[t+1].cell,
                                           action_t,
                                           env_state_tp1,
                                           preds_tilde)
        ρ = π_prob/b_prob
        δ_all[t] =  offpolicy_tdloss_gvfn(Float32.(ρ),
                                          preds_t,
                                          Float32.(cumulants),
                                          Float32.(discounts),
                                          preds_tilde)
    end

    return mean(δ_all)

end

function update!(model,
                 horde::AbstractHorde,
                 opt, lu::BatchTD, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0; kwargs...)

    ℒ_gvfn = begin
        if contains_gvfn(chain)
            gvfn_idx = find_layers_with_eq(chain, (l)->l isa Flux.Recur && l.cell isa AbstractGVFRCell)
            if length(gvfn_idx) != 1
                throw("Multi-layer GVFN Not available")
            end
            _gvfn_loss!(chain[1:gvfn_idx[1]],
                        lu,
                        h_init,
                        state_seq,
                        env_state_tp1,
                        action_t,
                        b_prob;
                        kwargs...)
        else
            param(0)
        end
    end
    
    reset!(chain, h_init)
    preds = chain.(state_seq)
    cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    ℒ_out = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Flux.data(preds[end]))

    grads = Flux.Tracker.gradient(()->β*ℒ_out + (1-β)*ℒ_gvfn, Flux.params(chain))
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end

    # reset!(rnn, h_init)

    # rnn_out = rnn.(state_seq)
    # preds = out_model.(rnn_out)

    # δ_all = param(zeros(length(preds)-1))
    # for t in 1:(length(preds)-1)
    #     cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, preds[t+1].data)
    #     ρ = Float64.(π_prob./b_prob)
    #     δ_all[t] = mean(0.5.*tderror(preds[t], Float64.(cumulants), Float64.(discounts), preds[t+1].data).^2)
    # end

    # grads = Flux.Tracker.gradient(()->mean(δ_all), Flux.params(out_model, rnn))
    # reset!(rnn, h_init)
    # for weights in Flux.params(out_model, rnn)
    #     Flux.Tracker.update!(opt, weights, grads[weights])
    # end
end
