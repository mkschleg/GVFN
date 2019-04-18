module FluxUtils

using Flux, ArgParse

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
    return construct_runn(kt(parsed["cell"]), in, parsed[kt("num_hidden")], args...; kwargs...)
end

function construct_rnn(cell::AbstractString, in::Integer, num_hidden::Integer, args...; kwargs...)
    cell_func = getproperty(Flux, Symbol(cell))
    return cell_func(in, num_hidden, args...; kwargs...)
end

function get_activation(act::AbstractString)
    if act == "sigmoid"
        return Flux.σ
    elseif act == "tanh"
        return tanh
    elseif act == "linear"
        return Flux.identity
    else
        throw("$(act) not known...")
    end
end

end
