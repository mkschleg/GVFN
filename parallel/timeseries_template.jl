#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH -o TODO.out # Standard output
#SBATCH -e TODO.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

# This is a template file; fill out all the
# missing values and the assert before using
@assert false #TODO

const cfg_file = "TODO"
const user = "TODO"
const save_path = "TODO"

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/$(cfg_file)"; save_path="/home/$(user)/$(save_path)")
