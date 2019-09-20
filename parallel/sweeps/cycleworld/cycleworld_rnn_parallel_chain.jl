#!/usr/local/bin/julia

using Pkg
using Reproduce

Pkg.activate(".")

const save_loc = "cycleworld_rnn_chain_length"
const exp_file = "experiment/cycleworld_rnn.jl"
const exp_module_name = :CycleWorldRNNExperiment
const exp_func_name = :main_experiment
const optimizer = "Descent"
const alphas = 0.1 .* 1.5.^(-4:2:2)
const chain = 6:4:30
const truncation_percentages = 0.0:0.2:1.0
const truncations = [1, 2, 4, 6, 8, 10, 14, 18, 24, 30]

function make_arguments(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    cell = args["cell"]
    # truncation = Int64(floor(parse(Int64, args["chain"]) * parse(Float64, args["trunc_perc"])))
    truncation = args["truncation"]
    seed = args["seed"]
    chain = args["chain"]
    # save_file = "$(save_loc)/$(horde)/$(cell)/$(optimizer)_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--horde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--cell", cell, "--numhidden", chain, "--chain", chain, "--seed", seed]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "numworkers"
        arg_type=Int64
        default=1
        "--jobloc"
        arg_type=String
        default=joinpath(save_loc, "jobs")
        "--numjobs"
        action=:store_true
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict([
        "horde"=>["onestep"],
        "alpha"=>alphas,
        # "trunc_perc"=>truncation_percentages,
        "truncation"=>truncations,
        "cell"=>["RNN"],
        "chain"=>collect(chain),
        "seed"=>collect(1:10)
    ])
    arg_list = ["chain", "horde", "cell", "alpha", "truncation", "seed"]

    static_args = ["--steps", "10", "--exp_loc", save_loc]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    if parsed["numjobs"]
        @info "This experiment has $(length(collect(args_iterator))) jobs."
        println(collect(args_iterator)[num_workers])
        exit(0)
    end

    experiment = Experiment(save_loc,
                            exp_file,
                            exp_module_name,
                            exp_func_name,
                            args_iterator)

    create_experiment_dir(experiment)
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment; num_workers=num_workers, job_file_dir=parsed["jobloc"])
    post_experiment(experiment, ret)
end


main()
