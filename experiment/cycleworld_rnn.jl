__precompile__(true)

module CycleWorldExperiment

import Flux.Tracker

import JLD2
import LinearAlgebra.Diagonal

# using GVFN: CycleWorld, step!, start!
using Flux
using GVFN: CycleWorld, step!, start!
using GVFN
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using DataStructures: CircularBuffer


const CWU = GVFN.CycleWorldUtils
const FLU = GVFN.FluxUtils

function arg_parse(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))

    #Experiment
    GVFN.exp_settings!(as)
    CWU.env_settings!(as)
    FLU.opt_settings!(as)

    # shared settings
    GVFN.rnn_arg_table!(as)
    
    return as
end

function main_experiment(args::Vector{String})
    as = arg_parse()
    parsed = parse_args(args, as)
    main_experiment(parsed)
end

function main_experiment(parsed::Dict)

    savefile = GVFN.save_setup(parsed)

    num_steps = parsed["steps"]
    seed = parsed["seed"]

    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])

    out_pred_strg = zeros(num_steps)
    out_err_strg = zeros(num_steps)

    #Construct agent
    fc = (state, action)->CWU.build_features_cycleworld(state)
    fs = 2

    ap = GVFN.RandomActingPolicy([1.0])
    out_horde = Horde([GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())])

    rnntype = getproperty(Flux, Symbol(parsed["cell"]))
    chain = Flux.Chain(rnntype(fs, parsed["numhidden"]),
                       Dense(parsed["numhidden"], length(out_horde)))



    agent = GVFN.FluxAgent(out_horde,
                           chain,
                           fc,
                           fs,
                           ap,
                           parsed;
                           rng=rng,
                           init_func=(dims...)->glorot_uniform(rng, dims...))

    callback(env, agent, (rew, term, s_tp1), (out_preds, action), step) = begin
        out_pred_strg[step] = Flux.data(out_preds)[1]
        out_err_strg[step] = out_pred_strg[step][1] - CWU.oracle(env, "onestep", parsed["gamma"])[1]
    end

    GVFN.continuous_experiment(env, agent, num_steps, parsed["verbose"], parsed["progress"], callback; rng=rng)
    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    GVFN.save_results(savefile, results, parsed["working"])
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end
