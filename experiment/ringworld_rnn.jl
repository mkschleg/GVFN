__precompile__(true)

module RingWorldExperiment

using GVFN: RingWorld, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter
# using FileIO
using JLD2
using Reproduce
using Random

# using Flux.Tracker: TrackedArray, TrackedReal, track, @grad

using DataStructures: CircularBuffer
RWU = GVFN.RingWorldUtils
const FLU = GVFN.FluxUtils

function results_synopsis(results, ::Val{true})
    rmse = sqrt.(mean(results["err"].^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[Int(floor(length(rmse)*0.8)):end]),
        "lc"=>mean(reshape(rmse, 1000, :); dims=1)[1,:]
    ])
end

results_synopsis(results, ::Val{false}) = results

function arg_parse(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))

    GVFN.exp_settings!(as)
    RWU.env_settings!(as)
    FLU.opt_settings!(as)
    FLU.rnn_settings!(as)

    RWU.horde_settings!(as, "out")

    return as
end

function construct_agent(parsed, rng=Random.GLOBAL_RNG)
    # out_horde = cwu.get_horde(parsed, "out")
    out_horde = RWU.get_horde(parsed, "out")
    ap = GVFN.RandomActingPolicy([0.5f0, 0.5f0])
    
    fc = if parsed["cell"] == "ARNN"
        RWU.StandardFeatureCreator()
    else
        RWU.StandardFeatureCreatorWithAction()
    end
    fs = JuliaRL.FeatureCreators.feature_size(fc)

    initf=(dims...)->glorot_uniform(rng, dims...)
    
    rnntype = getproperty(GVFN, Symbol(parsed["cell"]))
    chain = if rnntype == GVFN.ARNN
        Flux.Chain(rnntype(fs, 4, parsed["numhidden"]; init=initf),
                   Dense(parsed["numhidden"], 16, Flux.relu; initW=initf),
                   Dense(16, length(out_horde); initW=initf))
    else
        Flux.Chain(rnntype(fs, parsed["numhidden"]),
                   Dense(parsed["numhidden"], 16, Flux.relu; initW=initf),
                   Dense(16, length(out_horde); initW=initf))
    end

    agent = GVFN.FluxAgent(out_horde,
                           chain,
                           fc,
                           fs,
                           ap,
                           parsed;
                           rng=rng)
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

    # Construct Environment
    env = RingWorld(parsed["size"])

    agent = construct_agent(parsed, rng)
    # Out Horde

    out_pred_strg = zeros(num_steps, length(agent.horde))
    out_err_strg = zeros(num_steps, length(agent.horde))

    callback(env, agent, (rew, term, s_tp1), (out_preds, action), step) = begin
        out_pred_strg[step, :] .= Flux.data(out_preds)
        out_err_strg[step, :] .= out_pred_strg[step, :] .- RWU.oracle(env, parsed["outhorde"], parsed["outgamma"])
    end

    GVFN.continuous_experiment(env, agent, num_steps, parsed["verbose"], parsed["progress"], callback; rng=rng)

    results = Dict("err"=>out_err_strg, "pred"=>out_pred_strg)
    results = results_synopsis(results, Val(parsed["sweep"]))
    GVFN.save_results(savefile, results, parsed["working"])
end



end
