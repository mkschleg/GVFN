using Pkg

# cd("..")
Pkg.activate(".")
# include("parallel_experiment.jl")
# println("Hello Wolrd...")

using Reproduce

const save_loc = "final_ringworld_rnn_action"
const exp_file = "experiment/ringworld_action_rnn.jl"
const exp_module_name = :RingWorldRNNExperiment
const exp_func_name = :main_experiment


function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "--numworkers"
        arg_type=Int64
        default=4
        "--jobloc"
        arg_type=String
        default=joinpath(save_loc, "jobs")
        "--numjobs"
        action=:store_true
        "--numsteps"
        arg_type=Int64
        default=500000
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_list = [
        ["--truncation", "1", "--opt", "Descent", "--optparams", "0.00877915"],
        ["--truncation", "2", "--opt", "Descent", "--optparams", "0.15"],
        ["--truncation", "4", "--opt", "Descent", "--optparams", "0.1"],
        ["--truncation", "8", "--opt", "Descent", "--optparams", "0.15"],
        ["--truncation", "12", "--opt", "Descent", "--optparams", "0.1"],
        ["--truncation", "16", "--opt", "Descent", "--optparams", "0.1"]
    ]
    runs_iter = 6:(6+30)

    static_args = ["--steps", string(parsed["numsteps"]), "--exp_loc", save_loc]
    args_iterator = ArgLooper(arg_list, static_args, runs_iter, "--seed")

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
