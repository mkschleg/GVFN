#!/usr/local/bin/julia

using Pkg
import Reproduce: ArgIterator
# cd("..")
Pkg.activate(".")
include("parallel_experiment_new.jl")

# println("Hello Wolrd...")

#------ Optimizers ----------#
const save_loc = "compworld_rnn_sweep"

# Parameters for the SGD Algorithm
const optimizer = "ADAM"
const alphas = [0.0001]
const truncations = [10, 16, 24]
# const alphas = [0.0001, 0.001, 0.01]
# const alphas = [0.001,0.01,0.1]

# const alphas = 0.1*1.5.^(-6:1)

# # Parameters for the RMSProp Optimizer
# const optimizer = "RMSProp"
# const alphas = [0.0001, 0.001, 0.01]

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

function main()

    arg_dict = Dict{String, Any}()
    arg_list = Array{String, 1}()

    arg_dict = Dict([
        "horde"=>["forward"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "cell"=>["RNN", "LSTM", "GRU"],
        "seed"=>collect(1:2)
    ])
    arg_list = ["horde", "cell", "alpha", "truncation", "seed"]

    static_args = ["--steps", "50"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    create_experiment_dir(save_loc,
                          "experiment/compassworld_rnn.jl",
                          args_iterator;
                          exp_module_name=:CompassWorldRNNExperiment,
                          exp_func_name=:main_experiment
                          )
    job("experiment/compassworld_rnn.jl", args_iterator; exp_module_name=:CompassWorldRNNExperiment, exp_func_name=:main_experiment, num_workers=4)

end

main()
