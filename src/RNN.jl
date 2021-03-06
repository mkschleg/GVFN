
# Sepcifying a action-conditional RNN Cell

using Flux

#####
#
# Utilities for using RNNs and Action RNNs online and in chains.
#
#####

"""
    dont_learn_initial_state!

Simple function which prevents the initial hidden state stored by Flux.Recur to be not learnable. This isn't used in the code base as the hidden state is managed externally from the RNN, but could be useful later.
"""
dont_learn_initial_state!(rnn) = prefor(x -> x isa Flux.Recur && _dont_learn_initial_state_!(x), rnn)
dont_learn_initial_state_!(rnn) = 
    rnn.init = Flux.data(rnn.init)
function _dont_learn_initial_state!(rnn::Flux.Recur{Flux.LSTMCell})
    rnn.init = Flux.data.(rnn.init)
end

"""
    reset!(model, h_init)

Reset the hidden state of the model to h_init. If h_init is an IdDict the key will be the Flux.Recur layer (which is compatible w/ `get_hidden_state`.)
"""
reset!(m, h_init) = 
    Flux.prefor(x -> x isa Flux.Recur && _reset!(x, h_init), m)

reset!(m, h_init::IdDict) = 
    Flux.prefor(x -> x isa Flux.Recur && _reset!(x, h_init[x]), m)

function _reset!(m, h_init)
    Flux.reset!(m)
    if m.state isa Tuple
        if ndims(h_init[1]) == 2
            m.state = (param(copy(h_init[1])), param(copy(h_init[2])))
        else
            m.state[1].data .= Flux.data(h_init[1])
            m.state[2].data .= Flux.data(h_init[2])
        end
    else
        if ndims(h_init) == 1
            m.state.data .= Flux.data(h_init)
        else
            m.state = param(h_init)
        end
    end
end

# function _reset!(m::Flux.Recur{T}, h_init) where {T<:Flux.LSTMCell}
#     Flux.reset!(m)
#     if ndims(h_init[1]) == 2
#         m.state = (param(copy(h_init[1])), param(copy(h_init[2])))
#     else
#         m.state[1].data .= Flux.data(h_init[1])
#         m.state[2].data .= Flux.data(h_init[2])
#     end
# end

function _reset!(m, h_init::T) where {T<:AbstractArray{Float32,2}}
    m.state = param(h_init)
end

"""
    contains_rnntype(m, rnn_type::Type)

Return a boolean if rnn_type is contained in the model (useful for determining if this is a GVFN or if this is an ActionRNN.)
"""
function contains_rnntype(m, rnn_type::Type)
    is_rnn_type = Bool[]
    Flux.prefor(x -> push!(is_rnn_type, x isa Flux.Recur && x.cell isa rnn_type), m)
    return any(is_rnn_type)
end

"""
    find_layers_with_eq(m, eq)

Find layer indecies that fulfill condition `eq(l)`, returns an array of Union{Int, Tuple} to determine layer indecies.
"""
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

"""
    needs_action_input(m)

Returns a boolean if there is a layer requiring explicit action input (i.e. a tuple).
"""
function needs_action_input(m)
    needs_action = Bool[]
    Flux.prefor(x -> push!(needs_action, _needs_action_input(x)), m)
    return any(needs_action)
end

_needs_action_input(m) = false
_needs_action_input(m::Flux.Recur{T}) where {T} = _needs_action_input(m.cell)

"""
    get_next_hidden_state(model, h_init, input)

Get the next hidden state of the model (returned as an array or as an IdDict see `get_hidden_state`)
"""
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
"""
    get_hidden_state(model)

Return the hidden state of the model stored in an IdDict.
"""
function get_hidden_state(c)
    h_state = IdDict()
    Flux.prefor(x -> x isa Flux.Recur && get!(h_state, x, get_hidden_state(x)), c)
    h_state
end

get_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(Flux.data(rnn.state))
get_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = deepcopy(Flux.data.(rnn.state))

"""
    get_initial_hidden_state(model)

Return initial hidden state (useful for episode restarts).
"""
function get_initial_hidden_state(c)
    h_state = IdDict()
    Flux.prefor(x -> x isa Flux.Recur && get!(h_state, x, get_initial_hidden_state(x)), c)
    h_state
end

get_initial_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(Flux.data(rnn.init))
get_initial_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = deepcopy(Flux.data.(rnn.init))

function _get_next_hidden_state(c, datum::Array{<:AbstractFloat, 1})
    p = c(datum)
    get_hidden_state(c), p
end

function get_hidden_states_and_preds(c, data)
    map((datum)->_get_next_hidden_state(c, datum), data)
end

abstract type AbstractActionRNN end

_needs_action_input(m::M) where {M<:AbstractActionRNN} = true


function (m::Flux.RNNCell)(h, x::AbstractArray{Int, 1})
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(sum(Wi[:, x]; dims=2)[:,1] .+ Wh*h .+ b)
  return h, h
end

function (m::Flux.RNNCell)(h, x::AbstractArray{Int, 2})
    σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
    o_i = sum(Wi[:, x]; dims=2)[:,1,:]
    h = σ.(o_i .+ Wh*h .+ b)
    return h, h
end


"""
    ARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.
"""
mutable struct ARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    Wx::A
    Wh::A
    b::V
    h::H
end

#= 
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
=#

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


