#!/usr/local/bin/julia

using Pkg
using Reproduce

Pkg.activate(".")

const save_loc = "cycleworld_rnn_sweep_sgd"
const exp_file = "experiment/cycleworld_rnn.jl"
const exp_module_name = :CycleWorldRNNExperiment
const exp_func_name = :main_experiment
const optimizer = "RMSProp"
const alphas = 0.0001
const chain = [6, 10, 20, 30, 40, 50, 60, 70, 80, 100]
const truncation_percentages = 0.0:0.1:1.0

function make_arguments(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    cell = args["cell"]
    truncation = args["truncation"]
    seed = args["seed"]
    # save_file = "$(save_loc)/$(horde)/$(cell)/$(optimizer)_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--horde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--cell", cell, "--seed", seed]
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
        "horde"=>["onestep", "chain", "gamma_chain"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "cell"=>["RNN", "LSTM", "GRU"],
        "seed"=>collect(1:10)
    ])
    arg_list = ["horde", "cell", "alpha", "truncation", "seed"]

    static_args = ["--steps", "300000", "--numhidden", "7", "--exp_loc", save_loc]
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
