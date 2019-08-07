using Pkg

# cd("..")
Pkg.activate(".")
# include("parallel_experiment.jl")
# println("Hello Wolrd...")

using Reproduce

const save_loc = "final_compassworld_rnn"
const exp_file = "experiment/compassworld_rnn.jl"
const exp_module_name = :CompassWorldRNNExperiment
const exp_func_name = :main_experiment

#------ Learning Updates -------#

const learning_update = "TD"

#------ Optimizers ----------#

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
        default=1000000
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_list = [
        ["--cell", "RNN", "--truncation", "1", "--opt", "Descent", "--optparams", "0.225", "--feature", "action"],
        ["--cell", "RNN", "--truncation", "4", "--opt", "Descent", "--optparams", "0.225", "--feature", "action"],
        ["--cell", "RNN", "--truncation", "8", "--opt", "Descent", "--optparams", "0.15", "--feature", "action"],
        ["--cell", "RNN", "--truncation", "16", "--opt", "Descent", "--optparams", "0.15", "--feature", "action"],
        ["--cell", "RNN", "--truncation", "24", "--opt", "Descent", "--optparams", "0.15", "--feature", "action"],
        ["--cell", "RNN", "--truncation", "32", "--opt", "Descent", "--optparams", "0.15", "--feature", "action"],
        ["--cell", "RNN", "--truncation", "48", "--opt", "Descent", "--optparams", "0.15", "--feature", "action"],
        ["--cell", "RNN", "--truncation", "64", "--opt", "Descent", "--optparams", "0.15", "--feature", "action"],
        ["--cell", "LSTM", "--truncation", "1", "--opt", "Descent", "--optparams", "0.3375", "--feature", "action"],
        ["--cell", "LSTM", "--truncation", "4", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "LSTM", "--truncation", "8", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "LSTM", "--truncation", "16", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "LSTM", "--truncation", "24", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "LSTM", "--truncation", "32", "--opt", "Descent", "--optparams", "0.3375", "--feature", "action"],
        ["--cell", "LSTM", "--truncation", "48", "--opt", "Descent", "--optparams", "0.3375", "--feature", "action"],
        ["--cell", "LSTM", "--truncation", "64", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "GRU", "--truncation", "1", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "GRU", "--truncation", "4", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "GRU", "--truncation", "8", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "GRU", "--truncation", "16", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "GRU", "--truncation", "24", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "GRU", "--truncation", "32", "--opt", "Descent", "--optparams", "0.3375", "--feature", "action"],
        ["--cell", "GRU", "--truncation", "48", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"],
        ["--cell", "GRU", "--truncation", "64", "--opt", "Descent", "--optparams", "0.50625", "--feature", "action"]
    ]
    runs_iter = 6:(6+20)

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
