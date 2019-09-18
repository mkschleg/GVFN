__precompile__(true)

module CycleWorldRNNATExperiment

using GVFN: CycleWorld, step!, start!
using GVFN
import Flux
import Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter
using FileIO
using Reproduce
using Random
using DataStructures: CircularBuffer

# include("utils/util.jl")
import GVFN.CycleWorldUtils
import GVFN.FluxUtils



function exp_settings(as::ArgParseSettings = ArgParseSettings())

    #Experiment
    GVFN.exp_settings!(as)

    #Cycle world specific settings
    CycleWorldUtils.env_settings!(as)
    CycleWorldUtils.horde_settings!(as, "aux")

    
    # RNN
    FluxUtils.rnn_settings!(as)
    FluxUtils.opt_settings!(as)

    return as
end

# build_features(s) = [1.0, s[1], 1-s[1]]

function main_experiment(args::Vector{String})

    as = exp_settings()
    parsed = parse_args(args, as)

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
    progress = parsed["progress"]
    verbose = parsed["verbose"]
    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])



    horde = CycleWorldUtils.get_horde("onestep", parsed["chain"], 0.0)
    aux_horde = CycleWorldUtils.get_horde(parsed, "aux", length(horde))
    fc = (state, action)->CycleWorldUtils.build_features_cycleworld(state)
    fs = 3
    ap = GVFN.RandomActingPolicy([1.0])
    
    # agent = CycleWorldRNNAgent(parsed)
    agent = GVFN.RNNAgent(GVFN.Horde([horde.gvfs; aux_horde.gvfs]), fc, fs, ap, parsed;
                          rng=rng,
                          init_func=(dims...)->glorot_uniform(rng, dims...))
    num_gvfs = length(horde)

    out_pred_strg = zeros(num_steps, num_gvfs)
    out_err_strg = zeros(num_steps, num_gvfs)

    at_pred_strg = zeros(num_steps, length(aux_horde))
    at_err_strg = zeros(num_steps, length(aux_horde))

    _, s_t = start!(env)
    action = start!(agent, s_t; rng=rng)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    for step in 1:num_steps

        _, s_tp1, _, _ = step!(env, action)
        out_preds, action = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step, :] = Flux.data(out_preds[1:num_gvfs])
        out_err_strg[step, :] = out_pred_strg[step, :] .- CycleWorldUtils.oracle(env, "onestep", 0.0)

        at_pred_strg[step, :] = Flux.data(out_preds[(num_gvfs+1):end])
        at_err_strg[step, :] = at_pred_strg[step, :] .- CycleWorldUtils.oracle(env, parsed["auxhorde"], parsed["auxgamma"])

        if verbose
            println("step: $(step)")
            println(env)
            # println(agent)
            println(out_preds)
            # println("preds: ", CycleWorldUtils.oracle(env, "onestep", parsed["gamma"]))
            # println("Agent rnn-state: ", agent.rnn.state)
        end

        if progress
           next!(prg_bar)
        end
    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg, "at_err_strg"=>at_err_strg, "at_pred"=>at_pred_strg])
    if !parsed["working"]
        save(savefile, results)
    else
        return results
    end
    # return results
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

