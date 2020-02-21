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
const data_root = TODO
const save_loc = TODO
# =================

function configJob(cfg::ConfigManager, dir::AbstractString, num_runs::Int; kwargs...)
    exp_module_name = cfg.config_dict["config"]["exp_module_name"]
    exp_file = cfg.config_dict["config"]["exp_file"]
    exp_func_name = cfg.config_dict["config"]["exp_func_name"]
    if Reproduce.IN_SLURM()
        if !isdir(joinpath(dir, "jobs"))
            mkdir(joinpath(dir, "jobs"))
        end
        if !isdir(joinpath(dir, "jobs", cfg.config_dict["save_path"]))
            mkdir(joinpath(dir, "jobs", cfg.config_dict["save_path"]))
        end
    end
    job(exp_file, dir, Config.iterator(cfg, num_runs);
        exp_module_name=Symbol(exp_module_name),
        exp_func_name=Symbol(exp_func_name),
        exception_dir = joinpath("except", cfg.config_dict["save_path"]),
        job_file_dir = joinpath(dir, "jobs", cfg.config_dict["save_path"]),
        kwargs...)
end

function main()
    cfg = ConfigManager(cfg_file, data_root)
    parse!(cfg,1)

    # setup the data directories
    nruns=cfg["args"]["nruns"]
    nparams=total_combinations(cfg)
    for idx=1:nparams
        for r=1:nruns
            parse!(cfg,idx,r)
        end
    end

    create_experiment_dir(save_loc; org_file=false)
    configJob(cfg, save_loc, nruns)
end

main()
