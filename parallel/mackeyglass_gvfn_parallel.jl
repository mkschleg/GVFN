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
const steps = 600000
const valSteps = 200000
const testSteps = 200000

#------ Learning Updates -------#

const learning_update = "BatchTD"


#------ Model ----------#

const batchsize = [32]

# Parameters for the SGD Algorithm
const model_opt = ["ADAM"]
const model_stepsize = [0.001]

#------ GVFN ------#
const gvfn_stepsize = [3e-5]
const γlo = [0.2]
const γhi = [0.95]
const num_gvfs = [128]
const gvfn_opt = ["Descent"]

function make_arguments_rtd(args::Dict)
    horizon=args["horizon"]
    batchsize=args["batchsize"]

    model_stepsize=args["model_stepsize"]
    model_opt=args["model_opt"]

    gvfn_stepsize=args["gvfn_stepsize"]
    gvfn_opt=args["gvfn_opt"]
    gamma_high=args["gamma_high"]
    gamma_low=args["gamma_low"]
    num_gvfs=args["num_gvfs"]

    seed = args["seed"]

    new_args=[
        "--horizon",horizon,
        "--batchsize",batchsize,

        "--model_stepsize",model_stepsize,
        "--model_opt",model_opt,

        "--gvfn_stepsize" , gvfn_stepsize,
        "--gvfn_opt" , gvfn_opt,
        "--gamma_high",γhi,
        "--gamma_low",γlo,
        "--num_gvfs",num_gvfs,

        "--seed",seed
    ]
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
        "horizon"=>12,
        "batchsize"=>batchsize,

        "model_stepsize"=>model_stepsize,
        "model_opt"=>model_opt,

        "gvfn_stepsize" => gvfn_stepsize,
        "gvfn_opt" => gvfn_opt,
        "gamma_high"=>γhi,
        "gamma_low"=>γlo,
        "num_gvfs"=>num_gvfs,

        "seed"=>collect(1:5)
    ])
    arg_list = collect(keys(arg_dict))

    static_args = ["--alg", learning_update,
                   "--steps", string(steps),
                   "--valSteps", string(valSteps),
                   "--testSteps", string(testSteps),
                   "--exp_loc", save_loc]
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
    ret = job(experiment; num_workers=num_workers)
    post_experiment(experiment, ret)

end


main()
