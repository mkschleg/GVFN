module FluxUtils

using ..Flux
using Reproduce

function rnn_settings!(as::ArgParseSettings)
    @add_arg_table as begin
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
        "--cell"
        help="Cell"
        default="RNN"
        "--numhidden"
        help="Number of hidden units in cell"
        arg_type=Int64
        default=6
    end
end

function opt_settings!(as::ArgParseSettings, prefix::AbstractString="")
    add_arg_table(as,
                  "--$(prefix)opt",
                  Dict(:help=>"Optimizer",
                       :default=>"Descent"),
                  "--$(prefix)optparams",
                  Dict(:help=>"Parameters",
                       :arg_type=>Float64,
                       :default=>[],
                       :nargs=>'+'))
end

function get_optimizer(parsed::Dict)
    kt = keytype(parsed)
    get_optimizer(parsed[kt("opt")], parsed[kt("optparams")])
end

function get_optimizer(opt_string::AbstractString, params)
    opt_func = getproperty(Flux, Symbol(opt_string))
    return opt_func(params...)
end


function construct_rnn(in::Integer, parsed::Dict, args...; kwargs...)
    kt = keytype(parsed)
    return construct_rnn(kt(parsed["cell"]), in, parsed[kt("numhidden")], args...; kwargs...)
end

function construct_rnn(cell::AbstractString, in::Integer, num_hidden::Integer, args...; kwargs...)
    cell_func = getproperty(Flux, Symbol(cell))
    return cell_func(in, num_hidden, args...; kwargs...)
end

function clip(a)
    clamp.(a, 0.0, 1.0)
end

function clip(a::TrackedArray)
    track(clip, a)
end
Flux.Tracker.@grad function clip(a)
    return clip(Flux.data(a)), Δ -> Tuple(Δ)
end

function get_activation(act::AbstractString)
    if act == "sigmoid"
        return Flux.σ
    elseif act == "tanh"
        return tanh
    elseif act == "linear"
        return Flux.identity
    elseif act == "clip"
        
    else
        throw("$(act) not known...")
    end
end

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T}
    return Flux.data(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T<:Flux.LSTMCell}
    return Flux.data.(rnn.cell(h_init, input)[1])
end

get_initial_hidden_state(rnn::Flux.Recur{T}) where {T} = Flux.data(rnn.state)
get_initial_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = Flux.data.(rnn.state)

end
