
using JLD2
using MinimalRLCore
import ProgressMeter
import Reproduce

function save_setup(parsed; save_dir_key="exp_loc", working=false, def_save_file="results.jld2")
    savefile = def_save_file
    if !working
        Reproduce.create_info!(parsed, parsed[save_dir_key]; filter_keys=["verbose", "working", "progress", save_dir_key])
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, savefile)
        if isfile(savefile)
            return nothing
        end
    end
    savefile
end

function continuous_experiment(env, agent, num_steps, verbose=false, progress=false, callback_func=nothing; rng=Random.GLOBAL_RNG)
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)
    
    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    for step in 1:num_steps

        s_tp1, rew, term = step!(env, action, rng)
        out_preds, action = step!(agent, s_tp1, rew, term, rng)

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
