




macro rnn_arg_table(as)
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

function gvfn_arg_table!(as)
    @add_arg_table as begin
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
        "--alg"
        help="Algorithm"
        default="TDLambda"
        "--params"
        help="Parameters"
        arg_type=Float64
        nargs='+'
        "--act"
        help="Activation function for GVFN"
        arg_type=String
        default="identity"
    end
end



macro opt_arg_table(as)
    @add_arg_table as begin
        "--opt"
        help="Optimizer"
        default="Descent"
        "--optparams"
        help="Parameters"
        arg_type=Float64
        default=[]
        nargs='+'
    end
end


