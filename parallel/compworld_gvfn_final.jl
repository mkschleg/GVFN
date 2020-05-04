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

# include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))

list_args = [
    Dict("horde"=>"chain", "truncaction"=>1, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"chain", "truncaction"=>2, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncaction"=>3, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncaction"=>4, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncaction"=>6, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncaction"=>8, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncaction"=>12, "alpha"=>0.225),
    Dict("horde"=>"chain", "truncaction"=>16, "alpha"=>0.225),
    
    Dict("horde"=>"gamma_chain", "truncaction"=>1, "alpha"=>0.1*1.5^-6),
    Dict("horde"=>"gamma_chain", "truncaction"=>2, "alpha"=>0.15),
    Dict("horde"=>"gamma_chain", "truncaction"=>3, "alpha"=>0.15),
    Dict("horde"=>"gamma_chain", "truncaction"=>4, "alpha"=>0.1),
    Dict("horde"=>"gamma_chain", "truncaction"=>6, "alpha"=>0.15),
    Dict("horde"=>"gamma_chain", "truncaction"=>8, "alpha"=>0.225),
    Dict("horde"=>"gamma_chain", "truncaction"=>12, "alpha"=>0.15),
    Dict("horde"=>"gamma_chain", "truncaction"=>16, "alpha"=>0.15),

    Dict("horde"=>"gammas_aj", "truncaction"=>1, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncaction"=>2, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncaction"=>3, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncaction"=>4, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncaction"=>6, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncaction"=>8, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncaction"=>12, "alpha"=>0.1*1.5^-5),
    Dict("horde"=>"gammas_aj", "truncaction"=>16, "alpha"=>0.1*1.5^-3),
    
]


shared_args = Dict(
    "steps"=>500000,
    "size"=>6,
    "outhorde"=>"onestep",
    "gamma"=>0.95,
    "opt"=>"Descent"
)

runs = 1:30
seeds = runs .+ 10

reproduce_config_experiment("configs/ringworld_rnn.toml"; save_path="/home/mkschleg/scratch/GVFN")
