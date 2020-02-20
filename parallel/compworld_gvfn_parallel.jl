#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")


using Reproduce

as = ArgParseSettings()
@add_arg_table as begin
    "config"
    arg_type=String
end
parsed = parse_args(as)

experiment = Experiment(parsed["config"])

create_experiment_dir(experiment; tldr="")
add_experiment(experiment; settings_dir="settings")
ret = job(experiment)
post_experiment(experiment, ret)
