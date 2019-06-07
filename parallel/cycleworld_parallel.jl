#!/usr/local/bin/julia

using Pkg
Pkg.activate(".")

using Reproduce

const save_loc = "cycleworld_gvfn_sweep"
const exp_file = "experiment/cycleworld.jl"
const exp_module_name = :CycleWorldExperiment
const exp_func_name = :main_experiment
const optimizer = "Descent"
# const alphas = [0.001; 0.1*1.5.^(-6:3)]
const alphas = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75]

# const learning_update = "RTD"
# const truncations = [1, 3, 6, 12, 24]

const learning_update = "TDLambda"
const lambdas = 0.0:0.1:0.9


function make_arguments(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    lambda = args["lambda"]
    act = args["activation"]
    seed = args["seed"]
    new_args=["--horde", horde, "--params", lambda, "--act", act, "--opt", optimizer, "--optparams", alpha, "--seed", seed]
    return new_args
end

function main()

    arg_dict = Dict([
        "horde"=>["gamma_chain"],
        "alpha"=>[alphas[end]],
        "lambda"=>collect(0.0:0.1:0.9),
        "activation"=>["sigmoid"],
        "seed"=>collect(1:5)
    ])
    arg_list = [ "activation", "horde", "alpha", "lambda", "seed"]


    static_args = ["--steps", "200000", "--alg", learning_update, "--exp_loc", save_loc]
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
