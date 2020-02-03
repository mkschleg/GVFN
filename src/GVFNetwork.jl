using Lazy
using Flux

abstract type AbstractGVFRCell end

contains_gvfn(m) = contains_rnntype(m, AbstractGVFRCell)


"""
    GVFRCell

    A wrapper for RNN cells which
"""

mutable struct GVFRCell{R, H<:AbstractHorde} <: AbstractGVFRCell
    rnn::R
    horde::H
end

_needs_action_input(m::GVFRCell) = _needs_action_input(m.rnn)

(m::GVFRCell)(h, x) = m.rnn(h, x)
Flux.hidden(m::GVFRCell) = Flux.hidden(m.rnn)
Flux.@treelike GVFRCell
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

# """
#     GVFRActionCell

#     An RNN cell which takes advantage of the ActionRNN idea and the GVFN idea. This seems necessairy for 

#     Figure for A-RNN with 3 actions.
#         O - Concatenate
#         X - Split by action

#           -----------------------------------
#          |     |--> W_1*[o_{t+1};h_t]-|      |
#          |     |                      |      |
#   h_t    |     |                      |      | h_{t+1}
# -------->|-O---X--> W_2*[o_{t+1};h_t]-X--------------->
#          | |   |                      |      |
#          | |   |                      |      |
#          | |   |--> W_3*[o_{t+1};h_t]-|      |
#           -|---------------------------------
#            | (o_{t+1}, a_t)
#            |
# """
# mutable struct GVFRActionCell{F, A, B, V, H<:AbstractHorde} <: AbstractGVFRCell
#     σ::F
#     Wx::A
#     Wh::A
#     b::B
#     h::V
#     horde::H
# end

# GVFRActionCell(in, num_actions, horde; init=Flux.glorot_uniform, σ_int=σ) = begin
#     num_gvfs = length(horde)
#     GVFRActionCell(
#         σ_int,
#         param(init(num_actions, num_gvfs, in)),
#         param(init(num_actions, num_gvfs, num_gvfs)),
#         param(Flux.zeros(num_actions, num_gvfs)),
#         param(Flux.zeros(num_gvfs)),
#         horde)
# end

# @forward GVFRActionCell.horde get

# _needs_action_input(m::GVFRActionCell) = true

# Flux.hidden(m::GVFRActionCell) = m.h
# Flux.@treelike GVFRActionCell
# GVFRAction(args...; kwargs...) = Flux.Recur(GVFRActionCell(args...; kwargs...))

# function (m::GVFRActionCell)(h, x::Tuple{I, A}) where {I<:Integer, A}
#     new_h = m.σ.(m.Wx[x[1], :, :]*x[2] .+ m.Wh[x[1], :, :]*h .+ m.b[x[1], :])
#     return new_h, new_h
# end

# function (m::GVFRActionCell)(h, x::Tuple{Array{<:Integer, 1}, A}) where {A}
#     if length(size(h)) == 1
#         new_h = m.σ.(
#             cat(collect((m.Wx[x[1][i], :, :]*x[2][:, i]) for i in 1:length(x[1]))...; dims=2) .+
#             cat(collect((m.Wh[x[1][i], :, :]*h) for i in 1:length(x[1]))...; dims=2) .+
#             m.b[x[2], :]')
#         return new_h, new_h
#     else
#         new_h = m.σ.(
#             cat(collect((m.Wx[x[1][i], :, :]*x[2][:, i]) for i in 1:length(x[1]))...; dims=2) .+
#             cat(collect((m.Wh[x[1][i], :, :]*h[:, i]) for i in 1:length(x[1]))...; dims=2) .+
#             m.b[x[2], :]')
#         return new_h, new_h
#     end
# end


