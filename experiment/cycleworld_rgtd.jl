__precompile__(true)

module CycleWorldRGTDExperiment

import Flux
import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

# using GVFN: CycleWorld, step!, start!
using GVFN: CycleWorld, step!, start!
using GVFN
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using DataStructures: CircularBuffer


# include("utils/util.jl")
CWU = GVFN.CycleWorldUtils
FLU = GVFN.FluxUtils

function arg_parse(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))

    #Experiment
    
    GVFN.exp_settings!(as)
    CWU.env_settings!(as)
    FLU.opt_settings!(as)

    # shared settings
    GVFN.gvfn_arg_table!(as)

    @add_arg_table as begin
        "--horde"
        help="The horde used for training"
        default="gamma_chain"
        "--gamma"
        help="The gamma value for the gamma_chain horde"
        arg_type=Float64
        default=0.9
    end
    return as
end


function main_experiment(args::Vector{String})

    as = arg_parse()
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
    verbose = parsed["verbose"]
    progress = parsed["progress"]
    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])

    out_pred_strg = zeros(num_steps)
    out_err_strg = zeros(num_steps)

    _, s_t = start!(env)

    horde = CWU.get_horde(parsed)
    out_horde = Horde([GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())])
    fc = (state, action)->CWU.build_features_cycleworld(state)
    fs = 3
    ap = GVFN.RandomActingPolicy([1.0])
    
    agent = GVFN.RGTDAgent(horde, out_horde,
                           fc, fs, ap, parsed;
                           rng=rng,
                           # init_func=(dims...)->glorot_uniform(rng, dims...))
                           init_func=(dims...)->Flux.zeros(dims...).+(1//2))
    start!(agent, s_t; rng=rng)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    for step in 1:num_steps
 
        _, s_tp1, _, _ = step!(env, 1)
        out_preds, action, preds = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step] = Flux.data(out_preds)[1]
        out_err_strg[step] = out_pred_strg[step][1] - CWU.oracle(env, "onestep", parsed["gamma"])[1]

        if verbose
            println("step: $(step)")
            println(env)
            println(preds)
            println(out_preds)
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
