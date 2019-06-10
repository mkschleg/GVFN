#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o comp_gvfn_tdlambda.out # Standard output
#SBATCH -e comp_gvfn_tdlambda.err # Standard error
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

const save_loc = "compassworld_gvfn_tdlambda"
const exp_file = "experiment/compassworld.jl"
const exp_module_name = :CompassWorldExperiment
const exp_func_name = :main_experiment

#------ Learning Updates -------#

# const learning_update = "RTD"
# const truncations = [1, 5, 10, 16, 24, 32]


const learning_update = "TDLambda"
const lambdas = 0.0:0.1:0.9
# const truncations = [1, 10, 24]


#------ Optimizers ----------#

# Parameters for the SGD Algorithm
const optimizer = "Descent"
const alphas = clamp.(0.1*1.5.^(-6:4), 0.0, 1.0)
# const alphas = 0.1*1.5.^(-6:1)


function make_arguments_rtd(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    truncation = args["truncation"]
    seed = args["seed"]
    new_args=["--horde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--seed", seed]
    return new_args
end

function make_arguments_tdlambda(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    lambda = args["lambda"]
    seed = args["seed"]
    new_args=["--horde", horde, "--params", lambda, "--opt", optimizer, "--optparams", alpha, "--seed", seed]
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
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict{String, Any}()
    arg_list = Array{String, 1}()

    if learning_update == "RTD"
        arg_dict = Dict([
            "horde"=>["rafols", "forward"],
            "alpha"=>alphas,
            "truncation"=>truncations,
            "seed"=>collect(1:5)
        ])
        arg_list = ["horde", "alpha", "truncation", "seed"]
    elseif learning_update == "TDLambda"
        arg_dict = Dict([
            "horde"=>["rafols", "forward"],
            "alpha"=>alphas,
            "lambda"=>lambdas,
            "seed"=>collect(1:5)
        ])
        arg_list = ["horde", "alpha", "lambda", "seed"]
    end

    static_args = ["--alg", learning_update, "--steps", "2000000", "--exp_loc", save_loc]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=(learning_update == "RTD" ? make_arguments_rtd : make_arguments_tdlambda))

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
    ret = job(experiment; num_workers=4)
    post_experiment(experiment, ret)

end


main()
