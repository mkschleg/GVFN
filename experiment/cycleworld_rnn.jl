__precompile__(true)

module CycleWorldRNNExperiment

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

function Flux.Optimise.apply!(o::Flux.RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(Flux.data(x))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + Flux.Optimise.ϵ)
end

function exp_settings(as::ArgParseSettings = ArgParseSettings())

    #Experiment
    @add_arg_table as begin
        "--exp_loc"
        help="Location of experiment"
        arg_type=String
        default="tmp"
        "--seed"
        help="Seed of rng"
        arg_type=Int64
        default=0
        "--steps"
        help="number of steps"
        arg_type=Int64
        default=100
        "--verbose"
        action=:store_true
        "--working"
        action=:store_true
    end

    #Cycle world specific settings
    CycleWorldUtils.env_settings!(as)
    CycleWorldUtils.horde_settings!(as)

    # RNN
    FluxUtils.rnn_settings!(as)
    FluxUtils.opt_settings!(as)

    return as
end

build_features(s) = [1.0, s[1], 1-s[1]]

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
    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])
    
    horde = CycleWorldUtils.get_horde(parsed)
    fc = (state, action)->CycleWorldUtils.build_features_cycleworld(state)
    fs = 3
    ap = GVFN.RandomActingPolicy([1.0])
    
    # agent = CycleWorldRNNAgent(parsed)
    agent = GVFN.RNNAgent(horde, fc, fs, ap, parsed;
                          rng=rng,
                          init_func=(dims...)->glorot_uniform(rng, dims...))
    num_gvfs = length(agent.horde)

    out_pred_strg = zeros(num_steps, num_gvfs)
    out_err_strg = zeros(num_steps, num_gvfs)

    _, s_t = start!(env)
    action = start!(agent, s_t; rng=rng)

    @showprogress 0.1 "Step: " for step in 1:num_steps
        if parsed["verbose"]
            if step % 1000 == 0
                print(step, "\r")
            end
        end
        _, s_tp1, _, _ = step!(env, action)
        out_preds, action = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step,:] = Flux.data(out_preds)
        out_err_strg[step, :] = out_pred_strg[step, :] .- CycleWorldUtils.oracle(env, parsed["horde"], parsed["gamma"])
    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
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

