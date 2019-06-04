#!/usr/local/bin/julia

using Pkg
using ArgParse
# import Reproduce: ArgIterator
using Reproduce

# cd("..")
Pkg.activate(".")
# include("parallel_experiment_new.jl")

# println("Hello Wolrd...")

#------ Optimizers ----------#
const save_loc = "compworld_rnn_sweep"
const exp_file = "experiment/compassworld_rnn.jl"
const exp_module_name = :CompassWorldRNNExperiment
const exp_func_name = :main_experiment



# Parameters for the SGD Algorithm
const optimizer = "Descent"
const alphas = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75]
# const truncations = [1, 10, 16, 24]
const truncations = [1, 3, 6, 12, 24]


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
        "cell"=>["LSTM", "GRU", "RNN"],
        # "cell"=>["RNN", "GRU"],
        "seed"=>collect(1:5)
    ])
    arg_list = ["horde", "alpha", "truncation", "seed", "cell"]

    static_args = ["--steps", "5000"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    experiment = Experiment(save_loc,
                            exp_file,
                            exp_module_name,
                            exp_func_name,
                            args_iterator)
    # create_experiment_dir(save_loc)
    # add_experiment(save_loc,
    #                exp_file,
    #                string(exp_module_name),
    #                string(exp_func_name),
    #                args_iterator;
    #                settings_dir = "settings",
    #                )

    # job(exp_file, args_iterator;
    #     exp_module_name=exp_module_name,
    #     exp_func_name=exp_func_name,
    #     num_workers=num_workers)

    create_experiment_dir(experiment)
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment; num_workers=num_workers)
    post_experiment(experiment, ret)
end

main()
