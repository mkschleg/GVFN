#!/usr/local/bin/julia

using Pkg
# cd("..")
Pkg.activate(".")
include("parallel_experiment.jl")

println("Hello Wolrd...")
function make_arguments(args::Dict{String, String})
    horde = args["horde"]
    alpha = args["alpha"]
    truncation = args["truncation"]
    seed = args["seed"]
    save_file = "compassworld_gvfn/$(horde)/RTD/Descent_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--horde", horde, "--truncation", truncation, "--opt", "Descent", "--optparams", alpha, "--seed", seed, "--savefile", save_file]
    return new_args
end


const alphas = 0.1*1.5.^(-6:1)

function main()

    arg_dict = Dict([
        "horde"=>["rafols", "forward"],
        "alpha"=>alphas,
        "truncation"=>[1,2,3,4,6,8,10,16,24]
        "seed"=>collect(1:5)
    ])

    arg_list = ["horde", "alpha", "truncation", "seed"]
    static_args = ["--alg", "RTD", "--steps", "5000000"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)
    parallel_experiment_args("experiment/compassworld.jl", args_iterator; exp_module_name=:CompassWorldExperiment, exp_func_name=:main_experiment, num_workers=12)

end


main()
