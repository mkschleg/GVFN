using Pkg

# cd("..")
Pkg.activate(".")
# include("parallel_experiment.jl")
# println("Hello Wolrd...")

using Reproduce

const save_loc = "final_ringworld_rnn"
const exp_file = "experiment/ringworld_rnn.jl"
const exp_module_name = :RingWorldRNNSansActionExperiment
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
        ["--cell", "RNN", "--truncation", "1", "--opt", "Descent", "--optparams", "0.0131687"],
        ["--cell", "RNN", "--truncation", "2", "--opt", "Descent", "--optparams", "0.0131687"],
        ["--cell", "RNN", "--truncation", "4", "--opt", "Descent", "--optparams", "0.0131687"],
        ["--cell", "RNN", "--truncation", "8", "--opt", "Descent", "--optparams", "0.0131687"],
        ["--cell", "RNN", "--truncation", "12", "--opt", "Descent", "--optparams", "0.0131687"],
        ["--cell", "RNN", "--truncation", "16", "--opt", "Descent", "--optparams", "0.0197531"],
        ["--cell", "RNN", "--truncation", "24", "--opt", "Descent", "--optparams", "0.0197531"],
        ["--cell", "RNN", "--truncation", "32", "--opt", "Descent", "--optparams", "0.0197531"],
        ["--cell", "LSTM", "--truncation", "1", "--opt", "Descent", "--optparams", "0.0296296"],
        ["--cell", "LSTM", "--truncation", "2", "--opt", "Descent", "--optparams", "0.0666667"],
        ["--cell", "LSTM", "--truncation", "4", "--opt", "Descent", "--optparams", "0.0444444"],
        ["--cell", "LSTM", "--truncation", "8", "--opt", "Descent", "--optparams", "0.0666667"],
        ["--cell", "LSTM", "--truncation", "12", "--opt", "Descent", "--optparams", "0.0666667"],
        ["--cell", "LSTM", "--truncation", "16", "--opt", "Descent", "--optparams", "0.0666667"],
        ["--cell", "LSTM", "--truncation", "24", "--opt", "Descent", "--optparams", "0.0666667"],
        ["--cell", "LSTM", "--truncation", "32", "--opt", "Descent", "--optparams", "0.0666667"],
        ["--cell", "GRU", "--truncation", "1", "--opt", "Descent", "--optparams", "0.0296296"],
        ["--cell", "GRU", "--truncation", "2", "--opt", "Descent", "--optparams", "0.0444444"],
        ["--cell", "GRU", "--truncation", "4", "--opt", "Descent", "--optparams", "0.0296296"],
        ["--cell", "GRU", "--truncation", "8", "--opt", "Descent", "--optparams", "0.0296296"],
        ["--cell", "GRU", "--truncation", "12", "--opt", "Descent", "--optparams", "0.0296296"],
        ["--cell", "GRU", "--truncation", "16", "--opt", "Descent", "--optparams", "0.0296296"],
        ["--cell", "GRU", "--truncation", "24", "--opt", "Descent", "--optparams", "0.0296296"],
        ["--cell", "GRU", "--truncation", "32", "--opt", "Descent", "--optparams", "0.0296296"]
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
