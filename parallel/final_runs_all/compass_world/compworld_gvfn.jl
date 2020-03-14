#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o final_comp_gvfn.out # Standard output
#SBATCH -e final_comp_gvfn.err # Standard error
#SBATCH --mem-per-cpu=5000M # Memory request of 2 GB
#SBATCH --time=12:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce
using JLD2


const save_loc = "/home/mkschleg/scratch/GVFN/final_runs_all/final_compworld_gvfn"
const exp_file = "experiment/compassworld.jl"
const exp_module_name = :CompassWorldExperiment
const exp_func_name = :main_experiment

# include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))


const shared_args = Dict(
    "size"=>8,
    "policy"=>"rafols",
    "steps"=>1000000,
    "out-horde"=>"forward",
    "opt"=>"Descent",
    "save_dir"=>joinpath(save_loc, "data"),
    "sweep"=>true
)

@load "final_run_params/compassworld/compassworld_gvfn_descent_all.jld2" args_list

const runs = 1:30
const seeds = runs .+ 10
args_iterator = ArgLooper(args_list, shared_args, seeds, "seed")
# args_iterator = ArgLooper(args_list, shared_args, seeds, "seed")

exp = Experiment(save_loc,
                 exp_file,
                 exp_module_name,
                 exp_func_name,
                 args_iterator)


create_experiment_dir(exp)
add_experiment(exp; settings_dir="settings")
ret = job(exp; num_workers=4)
post_experiment(exp, ret)
