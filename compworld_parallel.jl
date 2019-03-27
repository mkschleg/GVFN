#!/usr/local/bin/julia

using Pkg
# cd("..")
Pkg.activate(".")
include("parallel_experiment.jl")

println("Hello Wolrd...")

#------ Learning Updates -------#

const learning_update = "RTD"
const truncations = [1, 10, 24]

const learning_update = "TDLambda"
const lambdas = 0.0:0.1:0.9
# const truncations = [1, 10, 24]


#------ Optimizers ----------#

# Parameters for the SGD Algorithm
const optimizer = "Descent"
const alphas = [0.001,0.01,0.1,0.25,0.5]
# const alphas = 0.1*1.5.^(-6:1)

# # Parameters for the RMSProp Optimizer
# const optimizer = "RMSProp"
# const alphas = [0.0001, 0.001, 0.01]

function make_arguments_rtd(args::Dict{String, String})
    horde = args["horde"]
    alpha = args["alpha"]
    truncation = args["truncation"]
    seed = args["seed"]
    save_file = "compassworld_gvfn/$(horde)/RTD/$(oprimizer)_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--horde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--seed", seed, "--savefile", save_file]
    return new_args
end

function make_arguments_tdlambda(args::Dict{String, String})
    horde = args["horde"]
    alpha = args["alpha"]
    lambda = args["lambda"]
    seed = args["seed"]
    save_file = "compassworld_gvfn/$(horde)/RTD/$(oprimizer)_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--horde", horde, "--luparams", lambda, "--opt", optimizer, "--optparams", alpha, "--seed", seed, "--savefile", save_file]
    return new_args
end

function main()

    arg_dict = Dict([
        "horde"=>["rafols", "forward"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "seed"=>collect(1:2)
    ])
    arg_list = ["horde", "alpha", "truncation", "seed"]
    if learning_update == "TDLambda"
        arg_dict = Dict([
            "horde"=>["rafols", "forward"],
            "alpha"=>alphas,
            "lambda"=>lambdas,
            "seed"=>collect(1:2)
        ])
        arg_list = ["horde", "alpha", "lambda", "seed"]
    end

    
    static_args = ["--alg", learning_update, "--steps", "5000000"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=learning_update == "RTD" ? make_arguments_rtd : make_arguments_tdlambda)
    parallel_experiment_args("experiment/compassworld.jl", args_iterator; exp_module_name=:CompassWorldExperiment, exp_func_name=:main_experiment, num_workers=8)

end


main()
