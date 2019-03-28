#!/usr/local/bin/julia

using Pkg
# cd("..")
Pkg.activate(".")
include("parallel_experiment.jl")

println("Hello Wolrd...")

#------ Learning Updates -------#

# const learning_update = "RTD"
# const truncations = [1, 10, 24]

# const learning_update = "TDLambda"
# const lambdas = 0.0:0.1:0.9
# # const truncations = [1, 10, 24]


#------ Optimizers ----------#

# Parameters for the SGD Algorithm
const optimizer = "Descent"
const alphas = [0.001,0.01,0.1]
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
    save_file = "cycleworld_rnn/$(horde)/$(cell)/$(optimizer)_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--horde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--seed", seed, "--savefile", save_file]
    return new_args
end

function main()

    arg_dict = Dict{String, Any}()
    arg_list = Array{String, 1}()

    arg_dict = Dict([
        "horde"=>["onestep", "chain"],
        "alpha"=>alphas,
        "truncation"=>collect(1:8),
        "cell"=>["RNN", "LSTM", "GRU"],
        "seed"=>collect(1:2)
    ])
    arg_list = ["horde", "cell", "alpha", "truncation", "seed"]

    static_args = ["--steps", "200000"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)
    parallel_experiment_args("experiment/cycleworld_rnn.jl", args_iterator; exp_module_name=:CycleWorldRNNExperiment, exp_func_name=:main_experiment, num_workers=6)

end


main()
