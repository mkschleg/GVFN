#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o comp_rnn_rmsprop.out # Standard output
#SBATCH -e comp_rnn_rmsprop.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")


using Reproduce

const save_loc = "/home/mkschleg/scratch/GVFN/compassworld_rnn_rmsprop_forward"
const exp_file = "experiment/compassworld_rnn.jl"
const exp_module_name = :CompassWorldRNNExperiment
const exp_func_name = :main_experiment

#------ Learning Updates -------#

const truncations = [1, 4, 8, 12, 16, 24, 32]

#------ Optimizers ----------#

const optimizer = "RMSProp"
const alphas = clamp.(0.001*1.5.^(-6:2:8), 0.0, 1.0)

function make_arguments_rtd(args::Dict)
    cell = args["cell"]
    alpha = args["alpha"]
    truncation = args["truncation"]
    seed = args["seed"]
    new_args=["--cell", cell, "--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--seed", seed]
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
        "--startruns"
        arg_type=Int
        default=1
        "--endruns"
        arg_type=Int
        default=10
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
        "cell"=>["ARNN", "RNN", "GRU", "LSTM"],
        "alpha"=>alphas,
        "truncation"=>truncations,
        "seed"=>collect(parsed["startruns"]:parsed["endruns"])
    ])
    arg_list = ["cell", "alpha", "truncation", "seed"]

    static_args = ["--steps", string(parsed["numsteps"]), "--exp_loc", save_loc, "--sweep", "--policy", "rafols", "--outhorde", "forward", "--numhidden", "40"]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments_rtd)

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
