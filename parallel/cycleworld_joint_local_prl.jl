#!/home/mkschleg/opt/bin/julia
#SBATCH --array=101-360
#SBATCH -o joint_out/%A_%a.out # Standard output
#SBATCH -e joint_out/%A_%a.err # Standard error
#SBATCH --mem=2000M # Memory request of 2 GB
#SBATCH --time=01:00:00 # Running time of 1 hours
#SBATCH --account=def-whitem

using Pkg
Pkg.activate(".")

using Reproduce
using Logging

const save_loc = "cycleworld_joint_sweep"
const exp_file = "experiment/cycleworld_joint.jl"
const exp_module_name = :CycleWorldJointExperiment
const exp_func_name = :main_experiment
const optimizer = "ADAM"
# const alphas = [0.001; 0.1*1.5.^(-6:2:1); 0.1*1.5.^(1:2:3)]
const alphas = [0.01]
const betas = 0.0:0.2:1.0
# const truncations = [1, 2, 4, 6, 8, 10]
const truncations = [3, 5, 7, 9]

function make_arguments(args::Dict{String, String})
    horde = args["outhorde"]
    alpha = args["alpha"]
    beta = args["beta"]
    cell = args["cell"]
    truncation = args["truncation"]
    seed = args["seed"]
    new_args=["--outhorde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--beta", beta, "--cell", cell, "--seed", seed]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "numworkers"
        arg_type=Int64
        default=1
        "--numsteps"
        arg_type=Int64
        default=200000
        "--numjobs"
        action=:store_true
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]
    
    arg_dict = Dict([
        # "outhorde"=>["onestep", "chain"],
        "outhorde"=>["onestep", "chain", "gamma_chain"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "cell"=>["RNN"],
        "beta"=>betas,
        "seed"=>collect(1:5)
    ])
    arg_list = ["outhorde", "cell", "alpha", "beta", "truncation", "seed"]
    static_args = ["--steps", string(parsed["numsteps"]), "--exp_loc", save_loc, "--gvfnhorde", "gamma_chain", "--gvfngamma", "0.9"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    if parsed["numjobs"]
        @info "This experiment has $(length(collect(args_iterator))) jobs."
        println(collect(args_iterator)[num_workers])
        exit(0)
    end

    # experiment = Experiment(save_loc)
    experiment = Experiment(save_loc,
                            exp_file,
                            exp_module_name,
                            exp_func_name,
                            args_iterator)

    create_experiment_dir(experiment)
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment; num_workers=num_workers)
    post_experiment(experiment, ret)
end


main()
