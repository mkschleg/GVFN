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

# This is just a template for a throwaway entry point.
# Copy to new file and fill in TODO's
@assert false

const cfg = #TODO
const save_loc = joinpath(ENV["SLURM_SUBMIT_DIR"])
const nruns = 10

create_experiment_dir(save_loc; org_file=false)
config_job(cfg,
           save_loc,
           nruns)
