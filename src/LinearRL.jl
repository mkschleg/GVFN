module LinearRL

using LinearAlgebra


function TD!(weights, α, ρ, state_t, state_tp1, reward, gamma, terminal)
    if terminal
        δ = reward - dot(weights, state_t)
    else
        δ = reward + gamma*dot(weights, state_tp1) - dot(weights, state_t)
    end
    # δ = 0
    # δ = δ + rand()
    # return δ
    # for i in 1:length(weights)
    #     weights[i] += α*ρ*δ*state_t[i]
    # end
    weights .+= α*ρ*δ.*state_t
    # weights += α*ρ*δ*state_t
end

function TDLambda!(weights, traces, α, ρ, state_t, state_tp1, reward, γ_t, γ_tp1, λ, terminal)

    δ = reward + γ_tp1*dot(weights, state_tp1) - dot(weights, state_t)
    traces .= ρ*(γ_t*λ*traces + state_t)
    weights .+= α*δ*traces
end

function BatchTDC!(weights, h, α, β, ρ, s_t, s_tp1, r, γ, terminal)
    # Python code sample = (prev_phi.copy(), phi.copy(), action, reward, state, prev_state, pi[action]/mu[action])
    # td_error = (sample[3] + gamma*(sample[1].dot(weights_tdc_iwer)) - sample[0].dot(weights_tdc_iwer))
    # weights_tdc_iwer = weights_tdc_iwer + alpha*sample[-1]*(td_error*sample[0] - gamma*sample[1]*(sample[0].dot(h_tdc_iwer)))
    # h_tlndc_iwer = h_tdc_iwer + alpha_h*(sample[-1]*td_error - sample[0].dot(h_tdc_iwer))*sample[0]

    δ = r + γ.*(dot.(s_tp1, [weights])) - dot.(s_t, [weights])
    Δθ = α*sum(ρ.*(δ.*s_t - γ.*dot.(s_t, [h]).*s_tp1))
    Δh = β*sum(s_t.*(ρ.*δ - dot.(s_t, [h])))

    weights .+= Δθ
    h .+= Δh
end

function WISBatchTDC!(weights, h, α, β, ρ, s_t, s_tp1, r, γ, terminal)
    # Python code sample = (prev_phi.copy(), phi.copy(), action, reward, state, prev_state, pi[action]/mu[action])
    # td_error = (sample[3] + gamma*(sample[1].dot(weights_tdc_iwer)) - sample[0].dot(weights_tdc_iwer))
    # weights_tdc_iwer = weights_tdc_iwer + alpha*sample[-1]*(td_error*sample[0] - gamma*sample[1]*(sample[0].dot(h_tdc_iwer)))
    # h_tlndc_iwer = h_tdc_iwer + alpha_h*(sample[-1]*td_error - sample[0].dot(h_tdc_iwer))*sample[0]

    δ = r + γ.*(dot.(s_tp1, [weights])) - dot.(s_t, [weights])
    Δθ = α*sum(ρ.*(δ.*s_t - γ.*dot.(s_t, [h]).*s_tp1))
    Δh = β*sum(s_t.*(ρ.*δ - dot.(s_t, [h])))

    weights .+= Δθ./sum(ρ)
    h .+= Δh./sum(ρ)
end

function BatchTDC2!(weights, h, α, β, ρ, s_t, s_tp1, r, γ, terminal)
    # Python code sample = (prev_phi.copy(), phi.copy(), action, reward, state, prev_state, pi[action]/mu[action])
    # td_error = (sample[3] + gamma*(sample[1].dot(weights_tdc_iwer)) - sample[0].dot(weights_tdc_iwer))
    # weights_tdc_iwer = weights_tdc_iwer + alpha*sample[-1]*(td_error*sample[0] - gamma*sample[1]*(sample[0].dot(h_tdc_iwer)))
    # h_tlndc_iwer = h_tdc_iwer + alpha_h*(sample[-1]*td_error - sample[0].dot(h_tdc_iwer))*sample[0]

    δ = r + γ.*(dot.(s_tp1, [weights])) - dot.(s_t, [weights])
    Δθ = α*sum(ρ.*(δ.*s_t - γ.*dot.(s_t, [h]).*s_tp1))
    Δh = β*sum(s_t.*(ρ.*(δ - dot.(s_t, [h]))))

    weights .+= Δθ
    h .+= Δh
end

function BatchGTD2!(weights, h, α, β, ρ, s_t, s_tp1, r, γ, terminal)
    # Python code sample = (prev_phi.copy(), phi.copy(), action, reward, state, prev_state, pi[action]/mu[action])
    # td_error = (sample[3] + gamma*(sample[1].dot(weights_gtd2_iwer)) - sample[0].dot(weights_gtd2_iwer))
    # weights_gtd2_iwer = weights_gtd2_iwer + alpha*sample[-1]*(sample[0] - gamma*sample[1])*(sample[0].dot(h_gtd2_iwer))
    # h_gtd2_iwer = h_gtd2_iwer + alpha_h*(sample[-1]*td_error - sample[0].dot(h_gtd2_iwer))*sample[0]

    δ = r + γ.*(dot.(s_tp1, [weights])) - dot.(s_t, [weights])
    Δθ = α*sum(ρ.*(s_t - γ.*s_tp1)*(dot.(s_t, [h])))
    Δh = β*sum(s_t.*(ρ.*δ - dot.(s_t, [h])))

    weights .+= Δθ
    h .+= Δh
end

# function WISBatchTDC!(weights, h, α, β, ρ, s_t, s_tp1, r, γ, terminal)
#     # δ = r + γ.*(dot.(s_tp1, [weights])) - dot.(s_t, [weights])
#     # Δθ = α*sum(ρ.*(δ.*s_t - γ.*dot.(s_t, [h]).*s_tp1))
#     # Δh = β*sum(s_t.*(ρ.*δ - dot.(s_t, [h])))

#     # weights .+= Δθ
#     # h .+= Δh
# end

# function WISBatchGTD2!(weights, h, α, β, ρ, s_t, s_tp1, r, γ, terminal)

# end


end
