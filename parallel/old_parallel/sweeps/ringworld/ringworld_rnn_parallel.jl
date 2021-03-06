#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o ring_rnn.out # Standard output
#SBATCH -e ring_rnn.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=12:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce

const save_loc = "ringworld_rnn_sweep_sgd"
const exp_file = "experiment/ringworld_rnn.jl"
const exp_module_name = :RingWorldRNNSansActionExperiment
const exp_func_name = :main_experiment
const optimizer = "Descent"
const alphas = clamp.(0.1*1.5.^(-6:6), 0.0, 1.0)
# const truncations = [1, 2, 4, 8, 12, 16]
const truncations = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64]

function make_arguments(args::Dict)
    alpha = args["alpha"]
    cell = args["cell"]
    truncation = args["truncation"]
    seed = args["seed"]
    # save_file = "$(save_loc)/$(horde)/$(cell)/$(optimizer)_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--cell", cell, "--seed", seed]
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
        "--numsteps"
        arg_type=Int64
        default=300000
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict([
        "alpha"=>alphas,
        "truncation"=>truncations,
        # "cell"=>["RNN", "LSTM", "GRU"],
        "cell"=>["RNN", "LSTM", "GRU"],
        "seed"=>collect(1:5)
    ])
    arg_list = ["cell", "alpha", "truncation", "seed"]

    static_args = ["--steps", string(parsed["numsteps"]), "--numhidden", "14", "--exp_loc", save_loc]
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
