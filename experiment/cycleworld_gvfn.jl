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
    GVFN.gvfn_arg_table!(as)

    @add_arg_table! as begin
        "--horde"
        help="The horde used for GVFN training"
        default="gamma_chain"
        "--gamma"
        help="The gamma value for the gamma_chain horde"
        arg_type=Float64
        default=0.9
    end
    return as
end


function construct_agent(parsed, rng=RNG.GLOBAL_RNG)
    #Construct agent
    horde = CWU.get_horde(parsed)
    out_horde = Horde([GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())])
    fc = (state, action)->CWU.build_features_cycleworld(state)
    fs = 2
    ap = GVFN.RandomActingPolicy([1.0])

    initf=(dims...)->glorot_uniform(rng, dims...)

    chain = Flux.Chain(GVFN.GVFR(horde, Flux.RNNCell, fs, length(horde), Flux.sigmoid, init=initf),
                       Flux.data,
                       Dense(length(horde), length(out_horde), initW=initf))

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)

    GVFN.FluxAgent(out_horde,
                   chain,
                   opt,
                   τ, fc, fs, ap;
                   rng=rng,
                   init_func=(dims...)->glorot_uniform(rng, dims...))
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

    agent = construct_agent(parsed, rng)

    # GVFN.continuous_experiment(env, agent, num_steps, parsed["verbose"], parsed["progress"], callback; rng=rng)
    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    verbose = parsed["verbose"]
    progress = parsed["progress"]

    cur_step = 1
    
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        out_pred_strg[cur_step] = Flux.data(a.out_preds)[1]
        out_err_strg[cur_step] = out_pred_strg[cur_step][1] - CWU.oracle(env, "onestep", parsed["gamma"])[1]

        if progress
           ProgressMeter.next!(prg_bar)
        end

        cur_step += 1
    end
    
    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    GVFN.save_results(savefile, results, parsed["working"])
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end
