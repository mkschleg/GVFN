#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o comp_gvfn.out # Standard output
#SBATCH -e comp_gvfn.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=64
#SBATCH --account=rrg-whitem

using Pkg

Pkg.activate(".")

using Reproduce

const save_loc = "mackeyglass_gvfn"
const exp_file = joinpath(@__DIR__,"../experiment/mackeyglass.jl")
const exp_module_name = :MackeyGlassExperiment
const exp_func_name = :main_experiment

#------ Learning Updates -------#

const learning_update = "RTD"
const truncations = [1, 5, 10, 16, 24, 32]


# const learning_update = "TDLambda"
# const lambdas = 0.0:0.1:0.9
# const truncations = [1, 10, 24]


#------ Optimizers ----------#

# Parameters for the SGD Algorithm
const optimizer = "ADAM"
const alphas = collect(2.0.^(-15:1))

const max_exponents=[7]


function make_arguments_rtd(args::Dict)
    horizon=args["horizon"]
    max_exp=args["max-exponent"]
    alpha = args["alpha"]
    truncation = args["truncation"]
    seed = args["seed"]
    new_args=["--max-exponent",max_exp, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--seed", seed, "--horizon", horizon]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "--numworkers"
        arg_type=Int64
        default=1
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

    arg_dict = Dict([
        "max-exponent"=>max_exponents,
        "horizon"=>12,
        "alpha"=>alphas,
        "truncation"=>truncations,
        "seed"=>collect(1:5)
    ])
    arg_list = ["horizon", "max-exponent","alpha", "truncation", "seed"]

    static_args = ["--alg", learning_update, "--steps", "60000", "--exp_loc", save_loc]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments_rtd)

    println(collect(args_iterator)[num_workers])
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
