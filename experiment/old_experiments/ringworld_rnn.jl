__precompile__(true)

module RingWorldRNNSansActionExperiment

import Flux
import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

# using GVFN: CycleWorld, step!, start!
using GVFN: RingWorld, step!, start!
using GVFN
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using DataStructures: CircularBuffer

RWU = GVFN.RingWorldUtils
FLU = GVFN.FluxUtils

function arg_parse(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))


    #Experiment

    GVFN.exp_settings!(as)
    RWU.env_settings!(as)
    FLU.opt_settings!(as)

    # shared settings
    FLU.rnn_settings!(as)
    
    return as
end

function main_experiment(args::Vector{String})

    as = arg_parse()
    parsed = parse_args(args, as)
    parsed["prev_action_or_not"] = true

    savepath = ""
    savefile = ""
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, "results.jld2")
        if isfile(savefile)
            return
        end
    end

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    verbose = parsed["verbose"]
    progress = parsed["progress"]
    # gamma = parsed["gamma"]
    # gamma = 0.0
    rng = Random.MersenneTwister(seed)

    env = RingWorld(parsed["size"])

    out_pred_strg = zeros(num_steps, 2)
    out_err_strg = zeros(num_steps, 2)

    _, s_t = start!(env)

    out_horde = RWU.onestep()
    fc = RWU.StandardFeatureCreator()
    fs = JuliaRL.FeatureCreators.feature_size(fc)
    ap = GVFN.RandomActingPolicy([0.5, 0.5])
    
    agent = GVFN.RNNAgent(out_horde,
                          fc, fs, ap, parsed;
                          rng=rng,
                          init_func=(dims...)->glorot_uniform(rng, dims...))
    action = start!(agent, s_t; rng=rng)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    for step in 1:num_steps

        _, s_tp1, _, _ = step!(env, action)
        out_preds, action = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step, :] .= Flux.data(out_preds)
        out_err_strg[step, :] = out_pred_strg[step, :] .- RWU.oracle(env, "onestep")

        if verbose
            println(step)
            println(env)
            println(agent)
        end

        if progress
           next!(prg_bar)
        end
    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])

    if !parsed["working"]
        JLD2.@save savefile results
    else
        return results
    end

end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end
