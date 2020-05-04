__precompile__(true)

module RingWorldRGTDExperiment

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


RWU = GVFN.RingWorldUtils
FLU = GVFN.FluxUtils


function default_arg_dict()
    Dict{String,Any}(
        "seed" => 1,
        "steps" => 300000,
        "size" => 6,
        "opt" => "Descent",
        "truncation" => 4,

        "outhorde" => "gammas_term",
        "outgamma" => 0.9,
        
        "alpha" => 0.01,

        "lu" => "RGTD",
        "rgtd-alpha" => 0.1,
        "rgtd-beta" => 0.01,
        
        "act" => "sigmoid",
        "horde" => "chain",
        "gamma" => 0.9,

        "save_dir" => "cycleworld_rgtd")
end


function construct_agent(parsed::Dict, rng=Random.GLOBAL_RNG)

    horde = RWU.get_horde(parsed)
    num_gvfs = length(horde)
    # out_horde always onestep
    out_horde = RWU.onestep()
    
    # fc = (state, action)->[1.0f0, Float32(state[1]), 1.0f0 - Float32(state[1])]
    fc = RWU.StandardFeatureCreator()
    fs = feature_size(fc)
    ap = GVFN.RandomActingPolicy([0.5f0, 0.5f0])

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

    gvfn = GVFN.GradientGVFN_act(fs, horde, 2, act; initθ=initf)

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



    env = RingWorld(parsed["size"])
    agent = construct_agent(parsed, rng)

    # display(GVFN.is_cumulant_mat(agent.gvfn))
    
    s_t = start!(env)
    start!(agent, s_t, rng)
    # construct agent

    out_pred_strg = zeros(Float32, num_steps, length(agent.out_horde))
    out_err_strg = zeros(Float32, num_steps, length(agent.out_horde))
    pred_strg = zeros(Float32, num_steps, length(agent.gvfn.horde))
    pred_err_strg = zeros(Float32, num_steps, length(agent.gvfn.horde))
    
    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    cur_step = 1
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_pred_strg[cur_step, :] = Flux.data(a.out_preds)
        out_err_strg[cur_step, :] = out_pred_strg[cur_step, :] - RWU.oracle(env, "onestep", 0.0)
        pred_strg[cur_step, :] = a.preds
        pred_err_strg[cur_step, :] .= a.preds - RWU.oracle(env, parsed["horde"], parsed["gamma"])
        if progress
            isdefined(Main, :IJulia) && Main.IJulia.clear_output(true)
            next!(prg_bar, showvalues=[(:err, mean(out_err_strg[1:cur_step].^2)), (:preds, mean(sqrt.(mean(pred_err_strg[(cur_step-100<1 ? 1 : cur_step-100):cur_step, :].^2; dims=2)))), (:step, cur_step)])
        elseif verbose
            println(env)
            println(agent)
            println(RWU.oracle(env, parsed["horde"], parsed["gamma"]))
        end
        cur_step += 1
    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg, "gvfn-pred"=>pred_strg, "pred_err_strg"=>pred_err_strg])

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
