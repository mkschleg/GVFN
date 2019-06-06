#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o comp_joint.out # Standard output
#SBATCH -e comp_join.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=01:00:00 # Running time of 12 hours
#SBATCH --ntasks=8
#SBATCH --account=def-whitem

using Pkg
Pkg.activate(".")


using Reproduce
using Logging

const save_loc = "cycleworld_joint_sweep_tmp"
const exp_file = "experiment/cycleworld_joint.jl"
const exp_module_name = :CycleWorldJointExperiment
const exp_func_name = :main_experiment
const optimizer = "ADAM"
# const alphas = [0.001; 0.1*1.5.^(-6:2:1); 0.1*1.5.^(1:2:3)]
const alphas = [0.01]
const betas = 0.0:0.2:1.0
const truncations = [1, 2, 4, 6, 8, 10]

# const num_steps = 200000

function make_arguments(args::Dict)
    horde = args["outhorde"]
    alpha = args["alpha"]
    beta = args["beta"]
    cell = args["cell"]
    truncation = args["truncation"]
    seed = args["seed"]
    new_args=["--gvfnhorde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--cell", cell, "--seed", seed, "--beta", beta]
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
        default=2000
        "--jobloc"
        arg_type=String
        default=joinpath(save_loc, "jobs")
        "--numjobs"
        action=:store_true
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict([
        "outhorde"=>["onestep", "chain"],
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
    ret = job(experiment; num_workers=num_workers, job_file_dir=parsed["jobloc"])
    post_experiment(experiment, ret)
end


main()
