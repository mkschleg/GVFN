#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o ring_rnn_rmsprop.out # Standard output
#SBATCH -e ring_rnn_rmsprop.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce
using JLD2

const save_loc = "final_ringworld_gvfn_rmsprop"
const exp_file = "experiment/ringworld.jl"
const exp_module_name = :RingWorldExperiment
const exp_func_name = :main_experiment

# include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))

const shared_args = Dict(
    "steps"=>50,
    "gamma"=>0.95,
    "opt"=>"RMSProp",
    "save_dir"=>joinpath(save_loc, "data"),
    "sweep"=>true
)

@load "final_run_params/ringworld/ringworld_gvfn_rmsprop.jld2" args_list

const runs = 1:30
const seeds = runs .+ 10
args_iterator = ArgLooper(args_list, shared_args, seeds, "seed")

exp = Experiment(save_loc,
                 exp_file,
                 exp_module_name,
                 exp_func_name,
                 args_iterator)


create_experiment_dir(exp)
add_experiment(exp; settings_dir="settings")
ret = job(exp; num_workers=4)
post_experiment(exp, ret)
