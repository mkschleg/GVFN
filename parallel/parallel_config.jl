
using Reproduce

function reproduce_config_experiment(config_file::AbstractString; tldr="", save_path="")
    experiment = Experiment(config_file, save_path)
    create_experiment_dir(experiment; tldr=tldr)
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment)
    post_experiment(experiment, ret)
end


function config_experiment(config_file, dir::AbstractString, runs::Int)
    create_experiment_dir(experiment)
    config_job(config_file, dir, runs)
end
