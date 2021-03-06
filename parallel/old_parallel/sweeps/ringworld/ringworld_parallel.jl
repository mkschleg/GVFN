#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o ring_gvfn_act.out # Standard output
#SBATCH -e ring_gvfn_act.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=64
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce

const save_loc = "ringworld_gvfn_sweep_sgd"
const exp_file = "experiment/ringworld_action.jl"
const exp_module_name = :RingWorldExperiment
const exp_func_name = :main_experiment
const optimizer = "Descent"
const alphas = clamp.(0.1*1.5.^(-6:1), 0.0, 1.0)


const learning_update = "RTD"
const truncations = [1, 2, 3, 4, 6, 8, 12, 16]

function make_arguments(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    trunc = args["truncation"]
    act = args["activation"]
    seed = args["seed"]
    new_args=["--horde", horde, "--truncation", trunc, "--act", act, "--opt", optimizer, "--optparams", alpha, "--seed", seed]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "numworkers"
        arg_type=Int64
        default=2
        "--jobloc"
        arg_type=String
        default=joinpath(save_loc, "jobs")
        "--numjobs"
        action=:store_true
        "--numsteps"
        arg_type=Int64
        default=500000
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict([
        #"horde"=>["chain", "gamma_chain", "gammas_aj_term"],
	# "horde"=>["half_chain", "gamma_chain"],
        "horde"=>["half_chain", "gamma_chain", "full_chain"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "activation"=>["relu", "sigmoid"],
        "seed"=>collect(1:5)
    ])
    arg_list = ["activation", "horde", "alpha", "truncation", "seed"]

    static_args = ["--steps", string(parsed["numsteps"]), "--alg", learning_update, "--exp_loc", save_loc]
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
