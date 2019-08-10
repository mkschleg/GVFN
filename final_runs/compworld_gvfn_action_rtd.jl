#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o comp_rnn_final.out # Standard output
#SBATCH -e comp_rnn_final.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=40
#SBATCH --account=rrg-whitem

using Pkg

# cd("..")
Pkg.activate(".")
# include("parallel_experiment.jl")
# println("Hello Wolrd...")

using Reproduce

const save_loc = "final_compassworld_gvfn_action_rtd"
const exp_file = "experiment/compassworld_action.jl"
const exp_module_name = :CompassWorldActionExperiment
const exp_func_name = :main_experiment

#------ Learning Updates -------#

const learning_update = "RTD"

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
        ["--act", "sigmoid","--horde", "rafols", "--truncation", "1", "--opt", "Descent", "--optparams", "0.1", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--truncation", "2", "--opt", "Descent", "--optparams", "0.08", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--truncation", "4", "--opt", "Descent", "--optparams", "0.05", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--truncation", "8", "--opt", "Descent", "--optparams", "0.05", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--truncation", "12", "--opt", "Descent", "--optparams", "0.05", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--truncation", "16", "--opt", "Descent", "--optparams", "0.05", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--truncation", "24", "--opt", "Descent", "--optparams", "0.05", "--feature", "standard"],
        ["--act", "sigmoid","--horde", "rafols", "--truncation", "32", "--opt", "Descent", "--optparams", "0.05", "--feature", "standard"]
    ]
    runs_iter = 6:(6+10)

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
