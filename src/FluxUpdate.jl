function _gvfn_loss!(chain,
                     h_init,
                     states,
                     env_state_tp1,
                     action_t=nothing,
                     b_prob=1.0;
                     kwargs...) where {H<:AbstractHorde}

    reset!(chain, h_init)
    preds = chain.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, π_prob = get(chain[end].cell,
                                       action_t,
                                       env_state_tp1,
                                       preds_tilde)
    ρ = π_prob/b_prob
    return offpolicy_tdloss_gvfn(Float32.(ρ),
                                 preds_t,
                                 Float32.(cumulants),
                                 Float32.(discounts),
                                 preds_tilde)

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

    grads = Flux.Tracker.gradient(()->ℒ_out + ℒ_gvfn, Flux.params(chain))
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end


function update!(chain,
                 horde::H,
                 opt,
                 lu::TD,
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

    grads = Flux.Tracker.gradient(()->ℒ_out + ℒ_gvfn, Flux.params(chain))
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end



function update!(chain,
                 horde::H,
                 opt,
                 lu::TD,
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

    grads = Flux.Tracker.gradient(()->ℒ_out + ℒ_gvfn, Flux.params(chain))
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end


struct TDInterpolate <: LearningUpdate
    β::Float32
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
