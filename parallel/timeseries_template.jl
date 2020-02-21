#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH -o TODO.out         # Standard output
#SBATCH -e TODO.err         # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=TODO         # Running time (hh:mm:ss)
#SBATCH --ntasks=TODO
#SBATCH --cpus-per-task=1
#SBATCH --account=rrg-whitem

using Pkg

Pkg.activate(".")

using Reproduce
using Reproduce.Config

# This is just a template for a
# throwaway entry point on the cluster
# don't use
@assert false #TODO

# === SET THESE ===
const cfg_file = TODO
const user = "ajjacobs"
# =================

# paths
const project_root = "/home/$(user)/GVFN"
const cfg_path = joinpath(project_roof,"configs/$(cfg_file)")
const data_path = joinpath(project_root,"data")

function configJob(cfg::ConfigManager, dir::AbstractString, num_runs::Int; kwargs...)
    exp_module_name = cfg.config_dict["config"]["exp_module_name"]
    exp_file = cfg.config_dict["config"]["exp_file"]
    exp_func_name = cfg.config_dict["config"]["exp_func_name"]
    if Reproduce.IN_SLURM()
        if !isdir(joinpath(dir, "jobs"))
            mkpath(joinpath(dir, "jobs"))
        end
        if !isdir(joinpath(dir, "jobs", cfg.config_dict["save_path"]))
            mkpath(joinpath(dir, "jobs", cfg.config_dict["save_path"]))
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
    cfg = ConfigManager(cfg_path, project_root)
    parse!(cfg,1)

    # setup the data directories
    nruns=cfg["args"]["nruns"]
    nparams=total_combinations(cfg)
    for idx=1:nparams
        for r=1:nruns
            parse!(cfg,idx,r)
        end
    end

    create_experiment_dir(data_path; org_file=false)
    configJob(cfg, data_path, nruns)
end

main()
