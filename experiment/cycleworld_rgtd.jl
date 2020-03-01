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

function default_arg_dict()
    Dict{String,Any}(
        "seed" => 1,
        "steps" => 100000,
        "chain" => 6,
        "opt" => "Descent",
        "truncation" => 4,

        "alpha" => 0.01,

        "lu" => "RGTD",
        "rgtd-alpha" => 0.3,
        "rgtd-beta" => 0.01,
        
        "act" => "sigmoid",
        "horde" => "chain",
        "gamma" => 0.9,

        "save_dir" => "cycleworld_rgtd")
end


function construct_agent(parsed::Dict, rng=Random.GLOBAL_RNG)

    horde = CWU.get_horde(parsed)
    num_gvfs = length(horde)
    # out_horde always onestep
    out_horde = CWU.onestep()
    
    fc = (state, action)->[1.0f0, Float32(state[1]), 1.0f0 - Float32(state[1])]
    fs = 3
    ap = GVFN.RandomActingPolicy([1.0])

    scale = get(parsed, "scale", 1.0f0)
    initf=(dims...)->scale*glorot_uniform(rng, dims...)

    lu_string = get(parsed, "alg", "RGTD")
    lu_type = getproperty(GVFN, Symbol(lu_string))
    @assert lu_type <: GVFN.AbstractGradUpdate
    lu = lu_type(parsed["rgtd-alpha"], parsed["rgtd-beta"])

    τ = parsed["truncation"]

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(parsed["alpha"])

    act = FluxUtils.get_activation(parsed["act"])

    gvfn = GradientGVFN(fs, horde, act; initθ=initf)

    num_out_gvfs = length(out_horde)
    model = Linear(num_gvfs, num_out_gvfs; init=initf)

    agent = GVFN.RGTDAgent(out_horde, # horde
                           gvfn, model, # models
                           lu, opt, τ, # optimization
                           fc, fs, # features
                           ap) # acting policy
end


function main_experiment(args::Vector{String})
    as = arg_parse()
    parsed = parse_args(args, as)
    main_experiment(parsed)
end

function main_experiment(parsed::Dict; verbose=false, working=false, progress=false)

    seed = parsed["seed"]
    num_steps = parsed["steps"]
    
    verbose = get(parsed, "verbose", verbose)
    progress = get(parsed, "progress", progress)
    working = get(parsed, "working", working)
    
    savefile = GVFN.save_setup(parsed; save_dir_key="save_dir", working=working)
    if savefile isa Nothing
        return
    end

    rng = Random.MersenneTwister(seed)



    env = CycleWorld(parsed["chain"])
    agent = construct_agent(parsed, rng)

    # display(GVFN.is_cumulant_mat(agent.gvfn))
    
    s_t = start!(env)
    start!(agent, s_t, rng)
    # construct agent

    out_pred_strg = zeros(Float32, num_steps)
    out_err_strg = zeros(Float32, num_steps)
    pred_err_strg = zeros(Float32, num_steps, length(agent.gvfn.horde))
    
    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    cur_step = 1
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_pred_strg[cur_step] = Flux.data(a.out_preds)[1]
        out_err_strg[cur_step] = out_pred_strg[cur_step][1] - CWU.oracle(env, "onestep", 0.0)[1]
        
        pred_err_strg[cur_step, :] .= a.preds - CWU.oracle(env, parsed["horde"], parsed["gamma"])
        if progress
            isdefined(Main, :IJulia) && Main.IJulia.clear_output(true)
            next!(prg_bar, showvalues=[(:err, mean(out_err_strg[1:cur_step].^2)), (:preds, mean(sqrt.(mean(pred_err_strg[(cur_step-100<1 ? 1 : cur_step-100):cur_step, :].^2; dims=2)))), (:step, cur_step)])
        end
        cur_step += 1
    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg, "pred_err_strg"=>pred_err_strg])

    if !working
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
