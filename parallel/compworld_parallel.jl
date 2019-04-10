#!/usr/local/bin/julia

using Pkg
using ArgParse
import Reproduce: ArgIterator

# cd("..")
Pkg.activate(".")
include("parallel_experiment_new.jl")

# println("Hello Wolrd...")

#------ Optimizers ----------#
const save_loc = "compworld_rnn_sweep"
const exp_file = "experiment/compassworld_rnn.jl"
const exp_module_name = :CompassWorldRNNExperiment
const exp_func_name = :main_experiment



# Parameters for the SGD Algorithm
const optimizer = "ADAM"
const alphas = [0.0001]
const truncations = [10, 16, 24]


function make_arguments(args::Dict{String, String})
    horde = args["horde"]
    alpha = args["alpha"]
    cell = args["cell"]
    truncation = args["truncation"]
    seed = args["seed"]
    save_file = "$(save_loc)/$(horde)/$(cell)/$(optimizer)_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--horde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--cell", cell, "--seed", seed, "--savefile", save_file]
    return new_args
end

function main(args::Vector{String}=ARGS)

    as = ArgParseSettings()
    @add_arg_table as begin
        "--numworkers"
        arg_type=Int64
        default=4
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict{String, Any}()
    arg_list = Array{String, 1}()

    arg_dict = Dict([
        "horde"=>["forward"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "cell"=>["LSTM"],
        # "cell"=>["RNN", "GRU"],
        "seed"=>collect(1:2)
    ])
    arg_list = ["horde", "alpha", "truncation", "seed", "cell"]

    static_args = ["--steps", "5000000"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)


    create_experiment_dir(save_loc)
    add_experiment(save_loc,
                   exp_file,
                   string(exp_module_name),
                   string(exp_func_name),
                   args_iterator;
                   settings_dir = "settings",
                   )

    job(exp_file, args_iterator;
        exp_module_name=exp_module_name,
        exp_func_name=exp_func_name,
        num_workers=num_workers)


end

main()
