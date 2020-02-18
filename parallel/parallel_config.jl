
using Reproduce

function reproduce_config_experiment(config_file::AbstractString; tldr="")
    experiment = Experiment(config_file)
    create_experiment_dir(experiment; tldr=tldr)
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment)
    post_experiment(experiment, ret)
end


function config_experiment(config_file, dir::AbstractString, runs::Int)
    create_experiment_dir(experiment; tldr=tldr)
    config_job(config_file, dir, runs)
end
