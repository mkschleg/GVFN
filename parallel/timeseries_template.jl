#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o TODO.out # Standard output
#SBATCH -e TODO.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=64
#SBATCH --account=rrg-whitem

using Pkg

Pkg.activate(".")

using Reproduce
using Reproduce.Config

# This is just a template for a
# throwaway entry point on the cluster
# don't use
@assert false

# === SET THESE ===
const cfg_file = TODO
const num_workers = TODO
# =================


const save_loc = joinpath(string(@__DIR__),"..")

cfg = ConfigManager(cfg_file, save_loc)
parse!(cfg, 1)

create_experiment_dir(save_loc; org_file=true)
config_job(cfg_file,
           save_loc,
           cfg["args"]["nruns"],
           num_workers=num_workers,
           extra_args=[save_loc]
           )
