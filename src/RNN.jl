
using Flux
import DataStructures

get_initial_hidden_state(rnn::Flux.Recur{T}) where {T} = Flux.data(rnn.state)
get_initial_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = Flux.data.(rnn.state)

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T}
    return Flux.data(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T<:Flux.LSTMCell}
    return Flux.data.(rnn.cell(h_init, input)[1])
end

mutable struct OnlineTD_RNN{T, H}
    τ::Integer
    states::DataStructures.CircularBuffer{T}
    hidden_state_strg::IdDict{Flux.Recur, H}
    OnlineTD_RNN(state_strg::DataStructures.CircularBuffer{T}, h_init::H) where {T, H} = new{T, H}(length(state_strg), state_strg, IdDict{Flux.Recur, H}())
end

function train_step!(out_model, rnn::Flux.Recur{T}, horde::AbstractHorde, opt, lu::OnlineTD_RNN, ϕ_tp1, env_state_tp1, action_t=nothing, b_prob=1.0) where {T}

    h_init = get!(lu.hidden_state_strg, rnn, get_initial_hidden_state(rnn))
    # h_init = get_initial_hidden_state(rnn)

    push!(lu.states, ϕ_tp1)

    reset!(rnn, h_init)
    rnn_out = rnn.(lu.states)
    preds = out_model.(rnn_out)

    cumulants, discounts, π_prob = get(horde, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Flux.data(preds[end]))

    grads = Flux.Tracker.gradient(()->δ, Flux.params(out_model, rnn))
    reset!(rnn, h_init)
    for weights in Flux.params(out_model, rnn)
        Flux.Tracker.update!(opt, weights, -grads[weights])
    end

    Flux.truncate!(rnn)
    rnn_out = rnn.(lu.states)
    preds = out_model.(rnn_out)

    lu.hidden_state_strg[rnn] = get_next_hidden_state(rnn, h_init, lu.states[1])

    return preds

end


struct OnlineJointTD{T, H}
    β::Float32
    states::DataStructures.CircularBuffer{T}
    hidden_state_strg::IdDict{Flux.Recur, H}
    OnlineJointTD(β::AbstractFloat, state_strg::DataStructures.CircularBuffer{T}, h_init::H) where {T, H} = new{T, H}(β, state_strg, IdDict{Flux.Recur, H}())
end

function train_step!(out_model, rnn::Flux.Recur{T}, horde::AbstractHorde, out_horde::AbstractHorde, opt, lu::OnlineJointTD, ϕ_tp1, env_state_tp1, action_t=nothing, b_prob=1.0) where {T}

    h_init = get!(lu.hidden_state_strg, rnn, get_initial_hidden_state(rnn))
    push!(lu.states, ϕ_tp1)

    reset!(rnn, h_init)
    rnn_out = rnn.(lu.states)

    preds = out_model.(rnn_out)

    cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(rnn_out[end]))
    ρ = Float32.(π_prob./b_prob)
    gvfn_δ = offpolicy_tdloss(ρ, rnn_out[end-1], Float32.(cumulants), Float32.(discounts), Float32.(Flux.data(rnn_out[end])))

    cumulants, discounts, π_prob = get(out_horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    out_δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Float32.(Flux.data(preds[end])))

    # grads = Flux.Tracker.gradient(()->(lu.β*out_δ + (1.0f0-lu.β)*gvfn_δ), Flux.params(rnn))
    δ = (lu.β*out_δ + ((1.0f0-lu.β)*gvfn_δ))
    grads = Flux.Tracker.gradient(()->δ, Flux.params(rnn))

    for weights in Flux.params(rnn)
        Flux.Tracker.update!(opt, weights, -grads[weights])
    end

    out_δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Float32.(Flux.data(preds[end])))

    grads = Flux.Tracker.gradient(()->(out_δ), Flux.params(out_model))

    for weights in Flux.params(out_model)
        Flux.Tracker.update!(opt, weights, -grads[weights])
    end

    reset!(rnn, h_init)
    Flux.truncate!(rnn)
    rnn_out = rnn.(lu.states)
    preds = out_model.(rnn_out)

    lu.hidden_state_strg[rnn] = get_next_hidden_state(rnn, h_init, lu.states[1])

    return preds, rnn_out

end

struct RTD_RNN{T, H}
    states::DataStructures.CircularBuffer{T}
    hidden_state_strg::IdDict{Flux.Recur, H}
    RTD_RNN(state_strg::DataStructures.CircularBuffer{T}, h_init::H) where {T, H} = new{T, H}(state_strg, IdDict{Flux.Recur, H}())
end

function train_step!(out_model, rnn::Flux.Recur{T}, horde::AbstractHorde, out_horde::AbstractHorde, opt, lu::RTD_RNN, ϕ_tp1, env_state_tp1, action_t=nothing, b_prob=1.0) where {T}

    h_init = get!(lu.hidden_state_strg, rnn, get_initial_hidden_state(rnn))
    push!(lu.states, ϕ_tp1)

    reset!(rnn, h_init)
    rnn_out = rnn.(lu.states)

    preds = out_model.(Flux.data.(rnn_out[end-1:end]))

    cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(rnn_out[end]))
    ρ = Float32.(π_prob./b_prob)
    gvfn_δ = offpolicy_tdloss(ρ, rnn_out[end-1], Float32.(cumulants), Float32.(discounts), Float32.(Flux.data(rnn_out[end])))

    cumulants, discounts, π_prob = get(out_horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    out_δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Float32.(Flux.data(preds[end])))

    grads = Flux.Tracker.gradient(()->(gvfn_δ), Flux.params(rnn))

    for weights in Flux.params(rnn)
        Flux.Tracker.update!(opt, weights, -grads[weights])
    end

    grads = Flux.Tracker.gradient(()->(out_δ), Flux.params(out_model))

    for weights in Flux.params(out_model)
        Flux.Tracker.update!(opt, weights, -grads[weights])
    end

    reset!(rnn, h_init)
    Flux.truncate!(rnn)
    rnn_out = rnn.(lu.states)
    preds = out_model.(rnn_out)

    lu.hidden_state_strg[rnn] = get_next_hidden_state(rnn, h_init, lu.states[1])

    return preds, rnn_out

end
