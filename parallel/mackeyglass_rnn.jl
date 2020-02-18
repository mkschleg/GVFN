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

#------ static parameters -------#

const save_loc = "mackeyglass/rnn"
const env_t = "MackeyGlass"
const agent_t = "RNN"
const exp_file = joinpath(@__DIR__,"../experiment/timeseries.jl")

const steps = 600000
const valSteps = 200000
const testSteps = 200000

const exp_module_name = :TimeSeriesExperiment
const exp_func_name = :main_experiment

const learning_update = "BatchTD"
const seeds = collect(1:10).+20437

#------ Model ------#

const batchsize = [32]
const rnn_cell = ["GRU"]

const rnn_opt = ["ADAM"]
const rnn_lr = [0.01, 0.005, 0.001, 0.0005, 0.0001]
const rnn_nhidden = [32,64,128]
const rnn_tau = [1,2,4,6,16,32]

function make_arguments(args::Dict)
    horizon = args["horizon"]
    batchsize = args["batchsize"]

    rnn_opt = args["rnn_opt"]
    rnn_lr = args["rnn_lr"]
    rnn_nhidden = args["rnn_nhidden"]
    rnn_tau = args["rnn_tau"]
    rnn_cell = args["rnn_cell"]

    seed = args["seed"]

    new_args=[
        "--horizon",horizon,
        "--batchsize",batchsize,

        "--rnn_opt",rnn_opt,
        "--rnn_lr",rnn_lr,
        "--rnn_nhidden", rnn_nhidden,
        "--rnn_tau", rnn_tau,
        "--rnn_cell", rnn_cell,

        "--seed",seed
    ]
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

    arg_dict = Dict([
        "horizon"=>12,
        "batchsize"=>batchsize,

        "rnn_opt"=>rnn_opt,
        "rnn_lr"=>rnn_lr,
        "rnn_nhidden"=>rnn_nhidden,
        "rnn_lr"=>rnn_lr,
        "rnn_tau"=>rnn_tau,
        "rnn_cell"=>rnn_cell,

        "seed"=>seeds
    ])
    arg_list = collect(keys(arg_dict))

    static_args = ["--alg", learning_update,
                   "--steps", string(steps),
                   "--valSteps", string(valSteps),
                   "--testSteps", string(testSteps),
                   "--exp_loc", save_loc,
                   "--agent", agent_t,
                   "--env", env_t]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

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
