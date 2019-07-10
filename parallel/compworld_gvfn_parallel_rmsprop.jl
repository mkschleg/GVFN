#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o comp_gvfn_rmsprop.out # Standard output
#SBATCH -e comp_gvfn_rmsprop.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=64
#SBATCH --account=rrg-whitem

using Pkg

# cd("..")
Pkg.activate(".")
# include("parallel_experiment.jl")
# println("Hello Wolrd...")

using Reproduce

const save_loc = "compassworld_gvfn_rmsprop"
const exp_file = "experiment/compassworld.jl"
const exp_module_name = :CompassWorldExperiment
const exp_func_name = :main_experiment

#------ Learning Updates -------#

const learning_update = "RTD"
const truncations = [1, 4, 8, 16, 24, 32, 48, 64]

#------ Optimizers ----------#

# Parameters for the SGD Algorithm
const optimizer = "RMSProp"
const alphas = 0.01*1.5.^(-8:3)



function make_arguments_rtd(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    truncation = args["truncation"]
    seed = args["seed"]
    feature = args["feature"]
    act = args["act"]
    new_args=["--horde", horde, "--act", act, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--feature", feature, "--seed", seed]
    return new_args
end

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

    arg_dict = Dict{String, Any}()
    arg_list = Array{String, 1}()

    
    arg_dict = Dict([
        "horde"=>["rafols", "forward", "gammas", "gammas_scaled"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "feature"=>["standard", "action"],
        "seed"=>collect(1:5),
        "act"=>["sigmoid", "relu"]
    ])
    arg_list = ["feature", "act", "horde", "alpha", "truncation", "seed"]


    static_args = ["--alg", learning_update, "--steps", string(parsed["numsteps"]), "--exp_loc", save_loc]
    args_iterator = ArgIterator(arg_dict, static_args;
                                arg_list=arg_list, make_args=make_arguments_rtd)

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
