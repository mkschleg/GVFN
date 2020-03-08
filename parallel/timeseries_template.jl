#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH -o TODO.out # Standard output
#SBATCH -e TODO.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem
#SBATCH --mail-user=ajjacobs@ualberta.ca
#SBATCH --mail-type=ALL

# This is a template file; fill out all the
# missing values and the assert before using
@assert false #TODO

# ================== SET THESE ===================
const user = "TODO"
const home = "/home/$(user)/"

const project_root = "/home/$(user)/TODO"
const save_dir = "/home/$(user)/TODO"

const cfg_path = "$(project_root)/configs/TODO"
# ================================================

using Pkg; Pkg.activate(project_root)
using Reproduce

function run(config_file::AbstractString; save_path="", num_workers=Inf)
    experiment = Experiment(config_file, save_path)
    create_experiment_dir(experiment)
    add_experiment(experiment; settings_dir="settings")
    ret = num_workers == Inf ? job(experiment) : job(experiment; num_workers=num_workers)
    post_experiment(experiment, ret)
end

run(cfg_path; save_path=save_dir)
