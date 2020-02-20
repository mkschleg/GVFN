#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH -o TODO.out # Standard output
#SBATCH -e TODO.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --account=rrg-whitem

using Pkg

Pkg.activate(".")

using Reproduce
using Reproduce.Config

# This is just a template for a
# throwaway entry point on the cluster
# don't use
@assert false

# === SET THIS ===
const cfg_file = TODO
# =================

const save_loc = joinpath(string(@__DIR__),"..")

nruns = parse!(ConfigManager(cfg_file, save_loc),1)["args"]["nruns"]

create_experiment_dir(save_loc; org_file=false)
config_job(cfg_file,
           save_loc,
           nruns)
