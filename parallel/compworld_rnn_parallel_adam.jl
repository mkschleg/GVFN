#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o comp_rnn_adam.out # Standard output
#SBATCH -e comp_rnn_adam.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce


#------ Optimizers ----------#
const save_loc = "compworld_rnn_sweep_adam"
const exp_file = "experiment/compassworld_rnn.jl"
const exp_module_name = :CompassWorldRNNExperiment
const exp_func_name = :main_experiment


# Parameters for the SGD Algorithm
const optimizer = "ADAM"
const alphas = 0.01*1.5.^(-8:2:2)
const truncations = [1, 5, 10, 16, 24, 32]

function make_arguments(args::Dict)
    horde = args["horde"]
    alpha = args["alpha"]
    cell = args["cell"]
    truncation = args["truncation"]
    feature = args["feature"]
    seed = args["seed"]
    new_args=["--horde", horde, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--cell", cell, "--feature", feature, "--seed", seed]
    return new_args
end

function main(args::Vector{String}=ARGS)

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

    arg_dict = Dict([
        "horde"=>["forward"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "cell"=>["RNN", "LSTM", "GRU"],
        "feature"=>["standard", "action"],
        # "cell"=>["RNN", "GRU"],
        "seed"=>collect(1:5)
    ])
    arg_list = ["feature", "horde", "alpha", "truncation", "seed", "cell"]

    static_args = ["--steps", "2000000", "--exp_loc", save_loc]
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
