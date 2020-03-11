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


const save_loc = "ringworld_gvfn_6_gammas_term"
const exp_file = "experiment/ringworld.jl"
const exp_module_name = :RingWorldExperiment
const exp_func_name = :main_experiment

# include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))

const args_list = [
    Dict("alpha" => 0.1*1.5^-5,"horde" => "chain","truncation" => 1),
    Dict("alpha" => 0.50625,"horde" => "chain","truncation" => 2),
    Dict("alpha" => 0.3375,"horde" => "chain","truncation" => 3),
    Dict("alpha" => 0.3375,"horde" => "chain","truncation" => 4),
    Dict("alpha" => 0.225,"horde" => "chain","truncation" => 6),
    Dict("alpha" => 0.3375,"horde" => "chain","truncation" => 8),
    Dict("alpha" => 0.225,"horde" => "chain","truncation" => 12),
    Dict("alpha" => 0.225,"horde" => "chain","truncation" => 16),

    Dict("alpha" => 0.1*1.5^-5, "horde" => "gamma_chain","truncation" => 1),
    Dict("alpha" => 0.3375,"horde" => "gamma_chain","truncation" => 2),
    Dict("alpha" => 0.3375,"horde" => "gamma_chain","truncation" => 3),
    Dict("alpha" => 0.225,"horde" => "gamma_chain","truncation" => 4),
    Dict("alpha" => 0.225,"horde" => "gamma_chain","truncation" => 6),
    Dict("alpha" => 0.225,"horde" => "gamma_chain","truncation" => 8),
    Dict("alpha" => 0.225,"horde" => "gamma_chain","truncation" => 12),
    Dict("alpha" => 0.225,"horde" => "gamma_chain","truncation" => 16),

    Dict("alpha" => 0.1*1.5^-1,"horde" => "gammas_aj","truncation" => 1),
    Dict("alpha" => 0.1*1.5^-4,"horde" => "gammas_aj","truncation" => 2),
    Dict("alpha" => 0.1*1.5^-5,"horde" => "gammas_aj","truncation" => 3),
    Dict("alpha" => 0.1*1.5^-5,"horde" => "gammas_aj","truncation" => 4),
    Dict("alpha" => 0.1*1.5^-5,"horde" => "gammas_aj","truncation" => 6),
    Dict("alpha" => 0.1*1.5^-5,"horde" => "gammas_aj","truncation" => 8),
    Dict("alpha" => 0.1*1.5^-5,"horde" => "gammas_aj","truncation" => 12),
    Dict("alpha" => 0.1*1.5^-5,"horde" => "gammas_aj","truncation" => 16),
]


const shared_args = Dict(
    "steps"=>50,
    "size"=>6,
    "outhorde"=>"gammas_term",
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
