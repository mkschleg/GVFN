#!/usr/local/bin/julia

using Pkg
# cd("..")
Pkg.activate(".")
include("parallel_experiment.jl")

println("Hello Wolrd...")
function make_arguments(args::Dict{String, String})
    horde = args["horde"]
    alpha = args["alpha"]
    lambda = args["lambda"]
    seed = args["seed"]
    save_file = "cycleworld_gvfn_julia/$(horde)/TDLambda_alpha_$(alpha)_lambda_$(lambda)_run_$(seed).jld2"
    new_args=["--horde", horde, "--params", alpha, lambda, "--seed", seed, "--savefile", save_file]
    return new_args
end

function main()

    arg_dict = Dict([
        # "horde"=>["chain", "gamma_chain"],
        "horde"=>["gamma_chain"],
        # "alpha"=>collect(0.1:0.1:0.9),
        "alpha"=>collect(0.5),
        "lambda"=>collect(0.0:0.1:0.9),
        "seed"=>collect(1:5)
    ])

    arg_list = ["horde", "alpha", "lambda", "seed"]
    static_args = ["--alg", "TDLambda", "--steps", "500000"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    parallel_experiment_args("experiment/cycleworld.jl", args_iterator; exp_module_name=:CycleWorldExperiment, exp_func_name=:main_experiment, num_workers=5)

    # for (args_idx, args) in args_iterator
    #     println(args_idx, " ", args)
    # end

end


main()
