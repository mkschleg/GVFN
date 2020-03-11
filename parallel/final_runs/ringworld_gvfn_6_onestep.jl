#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o ring_rnn.out # Standard output
#SBATCH -e ring_rnn.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce


const save_loc = "ringworld_gvfn_6_onestep"
const exp_file = "experiment/ringworld.jl"
const exp_module_name = :RingWorldExperiment
const exp_func_name = :main_experiment

# include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))

const args_list = [
    Dict("horde"=>"chain", "truncation"=>1, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"chain", "truncation"=>2, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncation"=>3, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncation"=>4, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncation"=>6, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncation"=>8, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncation"=>12, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncation"=>16, "alpha"=>0.225),
    
    Dict("horde"=>"gamma_chain", "truncation"=>1, "alpha"=>0.1*1.5^-6),
    Dict("horde"=>"gamma_chain", "truncation"=>2, "alpha"=>0.15),
    Dict("horde"=>"gamma_chain", "truncation"=>3, "alpha"=>0.15),
    Dict("horde"=>"gamma_chain", "truncation"=>4, "alpha"=>0.1),
    Dict("horde"=>"gamma_chain", "truncation"=>6, "alpha"=>0.15),
    Dict("horde"=>"gamma_chain", "truncation"=>8, "alpha"=>0.225),
    Dict("horde"=>"gamma_chain", "truncation"=>12, "alpha"=>0.15),
    Dict("horde"=>"gamma_chain", "truncation"=>16, "alpha"=>0.15),

    Dict("horde"=>"gammas_aj", "truncation"=>1, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncation"=>2, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncation"=>3, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncation"=>4, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncation"=>6, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncation"=>8, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncation"=>12, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncation"=>16, "alpha"=>0.1*1.5^-3),
]


const shared_args = Dict(
    "steps"=>50,
    "size"=>6,
    "outhorde"=>"onestep",
    "gamma"=>0.95,
    "opt"=>"Descent",
    "save_dir"=>joinpath(save_loc, "data")
)

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
# reproduce_config_experiment("configs/ringworld_rnn.toml"; save_path="/home/mkschleg/scratch/GVFN")
