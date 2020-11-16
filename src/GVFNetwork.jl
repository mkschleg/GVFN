using Lazy
using Flux

abstract type AbstractGVFRCell end

contains_gvfn(m) = contains_rnntype(m, AbstractGVFRCell)


"""
    GVFRCell

    A wrapper for RNN cells which contain a Flux RNN (or LSTM/GRU/other) cell and a horde for determining the loss of the GVFN.
"""
mutable struct GVFRCell{R, H<:AbstractHorde} <: AbstractGVFRCell
    rnn::R
    horde::H
end

_needs_action_input(m::GVFRCell) = _needs_action_input(m.rnn)

(m::GVFRCell)(h, x) = m.rnn(h, x)
Flux.hidden(m::GVFRCell) = Flux.hidden(m.rnn)
Flux.@treelike GVFRCell
get_hidden_state(rnn::Flux.Recur{GVFRCell{T, H}}) where {T<:Flux.LSTMCell, H} = deepcopy(Flux.data.(rnn.state))
get_initial_hidden_state(rnn::Flux.Recur{GVFRCell{T}}) where {T<:Flux.LSTMCell} = deepcopy(Flux.data.(rnn.init))


GVFR(args...; kwargs...) = Flux.Recur(GVFRCell(args...; kwargs...))

GVFR(horde::H, rnn_cell, args...; kwargs...) where {H<:AbstractHorde} = 
    Flux.Recur(GVFRCell(rnn_cell(args...; kwargs...), horde))

GVFR_RNN(in, horde, activation=σ; kwargs...) =
    Flux.Recur(GVFRCell(
        Flux.RNNCell(in, length(horde), activation; kwargs...), horde))

GVFR_GRU(in, horde; kwargs...) =
    Flux.Recur(GVFRCell(
        Flux.GRUCell(in, length(horde); kwargs...), horde))

GVFR_ARNN(in, num_actions, horde, activation=σ; init=Flux.glorot_uniform) =
    Flux.Recur(GVFRCell(
        ARNNCell(in, num_actions, length(horde), activation; init=init),
        horde))



@forward GVFRCell.horde get
const get_question_parameters = get
