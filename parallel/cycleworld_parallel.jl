#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o cycle_gvfn_lin.out # Standard output
#SBATCH -e cycle_gvfn_lin.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=1:00:00 # Running time of 12 hours
#SBATCH --ntasks=64
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce

const save_loc = "cycleworld_gvfn_sweep_sgd_lin"
const exp_file = "experiment/cycleworld.jl"
const exp_module_name = :CycleWorldExperiment
const exp_func_name = :main_experiment
const optimizer = "Descent"
const alphas = clamp.(0.1*1.5.^(-6:6), 0.0, 1.0)

# const learning_update = "RTD"
# const truncations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

const learning_update = "TDLambda"
const lambdas = 0.0:0.1:0.9


function make_arguments(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    lambda = args["lambda"]
    act = args["activation"]
    seed = args["seed"]
    new_args=["--horde", horde, "--params", lambda, "--act", act, "--opt", optimizer, "--optparams", alpha, "--seed", seed]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "numworkers"
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

    arg_dict = Dict([
        #"horde"=>["chain", "gamma_chain", "gammas_aj_term"],
	"horde"=>["gammas_aj_term"],
        "alpha"=>alphas,
        "lambda"=>lambdas,
        "activation"=>["sigmoid"],
        "seed"=>collect(1:10)
    ])
    arg_list = ["activation", "horde", "alpha", "lambda", "seed"]


    static_args = ["--steps", "300000", "--alg", learning_update, "--exp_loc", save_loc]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

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
