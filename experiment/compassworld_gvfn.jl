__precompile__(true)

module CompassWorldActionExperiment

using GVFN: CompassWorld, step!, start!
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
const cwu = GVFN.CompassWorldUtils

function results_synopsis(err, ::Val{true})
    rmse = sqrt.(mean(err.^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[Int64(floor(length(rmse)*0.8)):end]),
        "lc"=>mean(reshape(rmse, 1000, Int64(length(rmse)/1000)); dims=1)
    ])
end

results_synopsis(err, ::Val{false}) = sqrt.(mean(err.^2; dims=2))

function arg_parse(as::ArgParseSettings = ArgParseSettings())

    GVFN.exp_settings!(as)

    #Compass World
    @add_arg_table as begin
        "--policy"
        help="Acting policy of Agent"
        arg_type=String
        default="acting"
        "--size"
        help="The size of the compass world chain"
        arg_type=Int64
        default=8
    end

    # GVFN
    GVFN.agent_settings!(as, GVFN.GVFNActionAgent)

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

    # Construct Environment
    
    env = CompassWorld(parsed["size"], parsed["size"])

    # Construct Agent
    
    horde = cwu.get_horde(parsed)
    out_horde = cwu.get_horde(parsed, "out")

    # fc = cwu.StandardFeatureCreator()
    # if parsed["feature"] == "action"
    #     fc = cwu.ActionTileFeatureCreator()
    # end

    fc = cwu.NoActionFeatureCreator()

    fs = JuliaRL.FeatureCreators.feature_size(fc)

    ap = cwu.get_behavior_policy(parsed["policy"])

    chain = Flux.Chain(GVFN.GVFR(horde, GVFN.ARNNCell, fs, 3, length(horde), Flux.sigmoid),
                       Flux.data,
                       Dense(length(horde), length(out_horde)))

    agent = GVFN.FluxAgent(out_horde,
                           chain,
                           fc,
                           fs,
                           ap,
                           parsed;
                           rng=rng,
                           init_func=(dims...)->glorot_uniform(rng, dims...))

    out_pred_strg = zeros(num_steps, 5)
    out_err_strg = zeros(num_steps, 5)

    callback(env, agent, (rew, term, s_tp1), (out_preds, action), step) = begin
        out_pred_strg[step, :] .= Flux.data(out_preds)
        out_err_strg[step, :] .= out_pred_strg[step, :] .- cwu.oracle(env, parsed["outhorde"])
    end

    GVFN.continuous_experiment(env, agent, num_steps, parsed["verbose"], parsed["progress"], callback; rng=rng)

    results = results_synopsis(out_err_strg, Val(parsed["sweep"]))
    GVFN.save_results(savefile, results, parsed["working"])
end



end
