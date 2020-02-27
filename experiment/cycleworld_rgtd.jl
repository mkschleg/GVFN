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
    
    num_steps = parsed["steps"]
    seed = parsed["seed"]
    verbose = parsed["verbose"]
    progress = parsed["progress"]
    
    savefile = GVFN.save_setup(parsed; save_dir_key="save_dir", working=working)
    if savefile isa Nothing
        return
    end


    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])

    out_pred_strg = zeros(num_steps)
    out_err_strg = zeros(num_steps)


    s_t = start!(env)

    horde = CWU.get_horde(parsed)
    pred_err_strg = zeros(num_steps, length(horde))
    
    out_horde = Horde([GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())])
    fc = (state, action)->[1.0f0, Float32(state[1]), 1.0f0 - Float32(state[1])]
    fs = 3
    ap = GVFN.RandomActingPolicy([1.0])
    
    initf=(dims...)->glorot_uniform(rng, dims...)
    agent = GVFN.RGTDAgent(horde, out_horde,
                           fc, fs, ap, parsed;
                           rng=rng,
                           init_func=initf)

    println(GVFN.is_cumulant_mat(agent.gvfn))
    
    start!(agent, s_t, rng)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    cur_step = 1
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_pred_strg[cur_step] = Flux.data(a.out_preds)[1]
        out_err_strg[cur_step] = out_pred_strg[cur_step][1] - CWU.oracle(env, "onestep", 0.0)[1]
        
        pred_err_strg[cur_step, :] .= a.preds - CWU.oracle(env, parsed["horde"], parsed["gamma"])
        if progress
            next!(prg_bar, showvalues=[(:err, mean(out_err_strg[1:cur_step].^2)), (:preds, mean(sqrt.(mean(pred_err_strg[(cur_step-100<1 ? 1 : cur_step-100):cur_step, :].^2; dims=2)))), (:step, cur_step)])
        end
        cur_step += 1
    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg, "pred_err_strg"=>pred_err_strg])

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
