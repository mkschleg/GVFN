#!/usr/local/bin/julia

using Pkg
using Reproduce

Pkg.activate(".")

const save_loc = "cycleworld_rnn_sweep_test"
const exp_file = "experiment/cycleworld_rnn.jl"
const exp_module_name = :CycleWorldRNNExperiment
const exp_func_name = :main_experiment
const optimizer = "Descent"
const alphas = [0.001; 0.1*1.5.^(-6:2:1); 0.1*1.5.^(1:2:3)]
const truncations = [1, 2, 4, 6, 8, 10, 16]


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

    arg_dict = Dict([
        "horde"=>["onestep", "chain"],
        "alpha"=>[alphas[end]],
        "truncation"=>truncations,
        "cell"=>["RNN", "LSTM", "GRU"],
        # "cell"=>["RNN", "GRU"],
        "seed"=>collect(1:5)
    ])
    arg_list = ["horde", "cell", "alpha", "truncation", "seed"]

    static_args = ["--steps", "20"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    # experiment = Experiment(save_loc)
    experiment = Experiment(save_loc,
                            exp_file,
                            exp_module_name,
                            exp_func_name,
                            args_iterator)

    create_experiment_dir(experiment)
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment; num_workers=6)
    post_experiment(experiment, ret)
end


main()
