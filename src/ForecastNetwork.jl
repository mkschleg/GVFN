

"""
    TargetCell

A wrapper for RNN cells which ise used in FluxUpdate.jl. This forces the user to pass in explicit targets for the hidden state.
"""
mutable struct TargetCell{R}
    rnn::R
end

TargetR(rnn_cell, args...; kwargs...) = 
    Flux.Recur(TargetCell(rnn_cell(args...; kwargs...)))

TargetR_RNN(in, out, activation=tanh; kwargs...) =
    Flux.Recur(TargetCell(
        Flux.RNNCell(in, out, activation; kwargs...)))

TargetR_GRU(in, out; kwargs...) =
    Flux.Recur(TargetCell(
        Flux.GRUCell(in, out; kwargs...)))

TargetR_ARNN(in, num_actions, horde, activation=σ; init=Flux.glorot_uniform) =
    Flux.Recur(GVFRCell(
        ARNNCell(in, num_actions, length(horde), activation; init=init),
        horde))

Flux.hidden(m::TargetCell) = Flux.hidden(m.rnn)
Flux.@treelike TargetCell
(m::TargetCell)(h, x) = m.rnn(h, x)

get_hidden_state(rnn::Flux.Recur{TargetCell{T}}) where {T<:Flux.LSTMCell} = deepcopy(Flux.data.(rnn.state))
get_initial_hidden_state(rnn::Flux.Recur{TargetCell{T}}) where {T<:Flux.LSTMCell} = deepcopy(Flux.data.(rnn.init))




function _target_cell_loss(chain,
                           h_init,
                           hist_state_seq,
                           τ,
                           targets)
    reset!(chain, h_init)
    preds = chain.(hist_state_seq)
    Flux.mse(preds[τ], targets), preds
end

function _target_cell_batch_loss(chain, h_init, hist_state_seq, targets)
end
