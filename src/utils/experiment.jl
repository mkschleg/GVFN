
using JLD2
import ProgressMeter
import Reproduce

function save_setup(parsed, def_save_file="results.jld2")
    savefile = def_save_file
    if !(parsed["working"])
        Reproduce.create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, "results.jld2")
        if isfile(savefile)
            return nothing
        end
    end
    savefile
end

function continuous_experiment(env, agent, num_steps, verbose=false, progress=false, callback_func=nothing; rng=Random.GLOBAL_RNG)
    _, s_t = start!(env; rng=rng)
    action = start!(agent, s_t; rng=rng)
    
    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    for step in 1:num_steps

        _, s_tp1, rew, term = step!(env, action; rng=rng)
        out_preds, action = step!(agent, s_tp1, rew, term; rng=rng)

        if !(callback_func isa Nothing)
            callback_func(env, agent, (rew, term, s_tp1), (out_preds, action), step)
        end
        
        if verbose
            println(step)
            println(env)
            println(agent)
        end

        if progress
           ProgressMeter.next!(prg_bar)
        end
    end
end

function save_results(parsed::Dict, savefile, results)
    if !parsed["working"]
        JLD2.@save savefile results
    else
        return results
    end
end

function save_results(savefile, results, working=false)
    if !working
        JLD2.@save savefile results
    else
        return results
    end
end
