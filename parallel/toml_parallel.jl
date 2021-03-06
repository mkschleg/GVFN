#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o test_config.out # Standard output
#SBATCH -e test_config.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=00:15:00 # Running time of 12 hours
#SBATCH --ntasks=8
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "config"
        arg_type=String
        "--dest"
        arg_type=String
        default="./"
        "--numworkers"
        arg_type=Int
        default=4
        "--numjobs"
        action=:store_true
    end
    parsed = parse_args(as)
    
    experiment = Experiment(parsed["config"], parsed["dest"])

    create_experiment_dir(experiment; tldr="")
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment; num_workers=parsed["numworkers"])
    post_experiment(experiment, ret)
end

main()
