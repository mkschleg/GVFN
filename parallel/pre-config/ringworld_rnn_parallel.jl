#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
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

const save_loc = "/home/mkschleg/scratch/GVFN/ringworld_rnn_onestep"
const exp_file = "experiment/ringworld_rnn.jl"
const exp_module_name = :RingWorldExperiment
const exp_func_name = :main_experiment


const optimizer = "Descent"
const alphas = clamp.(0.1*1.5.^(-6:1), 0.0, 1.0)

# const learning_update = "RTD"
const truncations = [1, 2, 3, 4, 6, 8, 12, 16]
const rw_sizes = [6, 10]

function make_arguments(args::Dict)
    cell = args["cell"]
    alpha = args["alpha"]
    trunc = args["truncation"]
    seed = args["seed"]
    sze = args["size"]
    numhidden = if sze == "6"
        "14"
    elseif sze == "10"
        "22"
    end

    new_args=["--cell", cell, "--truncation", trunc, "--opt", optimizer, "--optparams", alpha, "--seed", seed, "--size", sze, "--numhidden", numhidden]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "numworkers"
        arg_type=Int
        default=5
        "--jobloc"
        arg_type=String
        default=joinpath(save_loc, "jobs")
        "--numjobs"
        action=:store_true
        "--startruns"
        arg_type=Int
        default=1
        "--endruns"
        arg_type=Int
        default=20
        "--numsteps"
        arg_type=Int
        default=300000
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]
    
    arg_dict = Dict(["cell"=>["ARNN", "RNN", "GRU", "LSTM"],
                     "alpha"=>alphas,
                     "truncation"=>truncations,
                     "size"=>rw_sizes,
                     "seed"=>collect(parsed["startruns"]:parsed["endruns"])])
    arg_list = ["size", "cell", "alpha", "truncation", "seed"]

    static_args = ["--outhorde", "onestep", "--steps", string(parsed["numsteps"]), "--exp_loc", save_loc, "--sweep"]
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
