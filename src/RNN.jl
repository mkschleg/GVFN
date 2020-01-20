
# Sepcifying a action-conditional RNN Cell

using Flux

# Utilities for using RNNs and Action RNNs online and in chains.

dont_learn_initial_state!(rnn) = prefor(x -> x isa Flux.Recur && _dont_learn_initial_state_!(x), rnn)
dont_learn_initial_state_!(rnn) = 
    rnn.init = Flux.data(rnn.init)
function _dont_learn_initial_state!(rnn::Flux.Recur{Flux.LSTMCell})
    rnn.init = Flux.data.(rnn.init)
end

reset!(m, h_init) = 
    Flux.prefor(x -> x isa Flux.Recur && _reset!(x, h_init), m)

reset!(m, h_init::IdDict) = 
    Flux.prefor(x -> x isa Flux.Recur && _reset!(x, h_init[x]), m)

function _reset!(m, h_init)
    Flux.reset!(m)
    m.state.data .= Flux.data(h_init)
end

function _reset!(m::Flux.Recur{T}, h_init) where {T<:Flux.LSTMCell}
    Flux.reset!(m)
    m.state[1].data .= Flux.data(h_init[1])
    m.state[2].data .= Flux.data(h_init[2])
end

function contains_rnntype(m, rnn_type::Type)
    is_rnn_type = Bool[]
    Flux.prefor(x -> push!(is_rnn_type, x isa Flux.Recur && x.cell isa rnn_type), m)
    return any(is_rnn_type)
end

function find_layers_with_eq(m, eq)
    layer_indecies = Union{Int, Tuple}[]
    for (idx, l) in enumerate(m)
        if l isa Flux.Chain
            layer_idx = find_layers_with_eq(l, eq)
            for l_idx in layer_idx
                push!(layer_indecies, (idx, l_idx))
            end
        elseif eq(l)
            push!(layer_indecies, idx)
        end
    end
    return layer_indecies
end

function needs_action_input(m)
    needs_action = Bool[]
    Flux.prefor(x -> push!(needs_action, _needs_action_input(x)), m)
    return any(needs_action)
end

_needs_action_input(m) = false
_needs_action_input(m::Flux.Recur{T}) where {T} = _needs_action_input(m.cell)

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T}
    return Flux.data(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T<:Flux.LSTMCell}
    return Flux.data.(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(c, h_init, input)
    reset!(c, h_init)
    c(input)
    get_hidden_state(c)
end

### TODO there may be extra copies here than what is actually needed. Test in the future if there is slowdown from allocations here.
function get_hidden_state(c)
    h_state = IdDict()
    Flux.prefor(x -> x isa Flux.Recur && get!(h_state, x, get_hidden_state(x)), c)
    h_state
end

get_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(Flux.data(rnn.state))
get_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = copy(Flux.data.(rnn.state))

function get_initial_hidden_state(c)
    h_state = IdDict()
    Flux.prefor(x -> x isa Flux.Recur && get!(h_state, x, get_initial_hidden_state(x)), c)
    h_state
end

get_initial_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(Flux.data(rnn.init))
get_initial_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = deepcopy(Flux.data.(rnn.init))


abstract type AbstractActionRNN end

_needs_action_input(m::M) where {M<:AbstractActionRNN} = true

"""
    ARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.

    Figure for A-RNN with 3 actions.
        O - Concatenate
        X - Split by action

          -----------------------------------
         |     |--> W_1*[o_{t+1};h_t]-|      |
         |     |                      |      |
  h_t    |     |                      |      | h_{t+1}
-------->|-O---X--> W_2*[o_{t+1};h_t]-X--------------->
         | |   |                      |      |
         | |   |                      |      |
         | |   |--> W_3*[o_{t+1};h_t]-|      |
          -|---------------------------------
           | (o_{t+1}, a_t)
           |
"""
mutable struct ARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    Wx::A
    Wh::A
    b::V
    h::H
end

ARNNCell(in, num_actions, out, activation=tanh; init=Flux.glorot_uniform) =
    ARNNCell(
        activation,
        param(init(num_actions, out, in)),
        param(init(num_actions, out, out)),
        param(zeros(Float32, num_actions, out)),
        param(Flux.zeros(out)))

function (m::ARNNCell)(h, x::Tuple{I, A}) where {I<:Integer, A}
    new_h = m.σ.(m.Wx[x[1], :, :]*x[2] .+ m.Wh[x[1], :, :]*h .+ m.b[x[1], :])
    return new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{Array{<:Integer, 1}, A}) where {A}
    if length(size(h)) == 1
        new_h = m.σ.(
            cat(collect((m.Wx[x[1][i], :, :]*x[2][:, i]) for i in 1:length(x[1]))...; dims=2) .+
            cat(collect((m.Wh[x[1][i], :, :]*h) for i in 1:length(x[1]))...; dims=2) .+
            m.b[x[2], :]')
        return new_h, new_h
    else
        new_h = m.σ.(
            cat(collect((m.Wx[x[1][i], :, :]*x[2][:, i]) for i in 1:length(x[1]))...; dims=2) .+
            cat(collect((m.Wh[x[1][i], :, :]*h[:, i]) for i in 1:length(x[1]))...; dims=2) .+
            m.b[x[2], :]')
        return new_h, new_h
    end
end

Flux.hidden(m::ARNNCell) = m.h
Flux.@treelike ARNNCell
ARNN(args...; kwargs...) = Flux.Recur(ARNNCell(args...; kwargs...))

function Base.show(io::IO, l::ARNNCell)
  print(io, "ARNNCell(", size(l.Wx, 2), ", ", size(l.Wx, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end


