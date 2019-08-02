using Pkg

# cd("..")
Pkg.activate(".")
# include("parallel_experiment.jl")
# println("Hello Wolrd...")

using Reproduce

const save_loc = "final_compassworld_gvfn_action_tdlambda"
const exp_file = "experiment/compassworld_action.jl"
const exp_module_name = :CompassWorldActionExperiment
const exp_func_name = :main_experiment

#------ Learning Updates -------#

const learning_update = "TDLambda"

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
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.0", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.1", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.2", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.3", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.4", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.5", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.6", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.7", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.8", "--opt", "Descent", "--optparams", "0.15", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--params", "0.9", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"]
    ]
    runs_iter = 6:(6+20)

    static_args = ["--alg", learning_update, "--steps", string(parsed["numsteps"]), "--exp_loc", save_loc]
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
