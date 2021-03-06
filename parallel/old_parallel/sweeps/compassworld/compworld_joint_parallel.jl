#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o comp_joint.out # Standard output
#SBATCH -e comp_joint.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=64
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce
using Logging

const save_loc = "compworld_joint_sweep"
const exp_file = "experiment/compworld_joint.jl"
const exp_module_name = :CompassWorldJointExperiment
const exp_func_name = :main_experiment
const optimizer = "ADAM"
# const alphas = [0.001; 0.1*1.5.^(-6:2:1); 0.1*1.5.^(1:2:3)]
const alphas = [0.01]
const betas = 0.0:0.2:1.0
const truncations = [1, 8, 16, 24, 32]

function make_arguments(args::Dict{String, String})
    horde = args["gvfnhorde"]
    alpha = args["alpha"]
    beta = args["beta"]
    cell = args["cell"]
    truncation = args["truncation"]
    seed = args["seed"]
    new_args=["--gvfnhorde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--beta", beta, "--cell", cell, "--seed", seed]
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
        default=2000000
        "--saveloc"
        arg_type=String
        default=save_loc
        "--numjobs"
        action=:store_true
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]
    
    arg_dict = Dict([
        # "outhorde"=>["onestep", "chain"],
        "gvfnhorde"=>["forward", "rafols", "gammas"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "cell"=>["RNN"],
        "beta"=>betas,
        "seed"=>collect(1:5)
    ])
    arg_list = ["gvfnhorde", "cell", "alpha", "beta", "truncation", "seed"]
    static_args = ["--steps", string(parsed["numsteps"]), "--exp_loc", parsed["saveloc"], "--outhorde", "forward", "--verbose"]
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
