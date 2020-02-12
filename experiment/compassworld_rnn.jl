__precompile__(true)

module CompassWorldRNNExperiment

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
const FLU = GVFN.FluxUtils

function results_synopsis(results, ::Val{true})
    rmse = sqrt.(mean(results["err"].^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[Int64(floor(length(rmse)*0.8)):end]),
        "lc"=>mean(reshape(rmse, 1000, Int64(length(rmse)/1000)); dims=1)
    ])
end

results_synopsis(results, ::Val{false}) = results

function arg_parse(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))

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

    # cwu.horde_settings!(as)
    cwu.horde_settings!(as, "out")

    # GVFN
    FLU.opt_settings!(as)
    FLU.rnn_settings!(as)

    return as
end

function construct_agent(parsed, rng=Random.GLOBAL_RNG)
    out_horde = cwu.get_horde(parsed, "out")
    ap = cwu.get_behavior_policy(parsed["policy"])
    
    fc = if parsed["cell"] == "ARNN"
        cwu.NoActionFeatureCreator()
    else
        cwu.StandardFeatureCreator()
    end
    fs = JuliaRL.FeatureCreators.feature_size(fc)

    initf=(dims...)->glorot_uniform(rng, dims...)
    rnntype = getproperty(GVFN, Symbol(parsed["cell"]))
    chain = if rnntype == GVFN.ARNN
        Flux.Chain(rnntype(fs, 4, parsed["numhidden"]; init=initf),
                           Dense(parsed["numhidden"], 32, Flux.relu; initW=initf),
                           Dense(32, length(out_horde); initW=initf))
    else
        Flux.Chain(rnntype(fs, parsed["numhidden"]; init=initf),
                           Dense(parsed["numhidden"], 32, Flux.relu; initW=initf),
                           Dense(32, length(out_horde); initW=initf))
    end

    agent = GVFN.FluxAgent(out_horde,
                           chain,
                           fc,
                           fs,
                           ap,
                           parsed;
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

    # Construct Environment
    env = CompassWorld(parsed["size"], parsed["size"])

    agent = construct_agent(parsed, rng)
    # Out Horde

    out_pred_strg = zeros(num_steps, length(agent.horde))
    out_err_strg = zeros(num_steps, length(agent.horde))

    callback(env, agent, (rew, term, s_tp1), (out_preds, action), step) = begin
        out_pred_strg[step, :] .= Flux.data(out_preds)
        out_err_strg[step, :] .= out_pred_strg[step, :] .- cwu.oracle(env, parsed["outhorde"])
    end

    GVFN.continuous_experiment(env, agent, num_steps, parsed["verbose"], parsed["progress"], callback; rng=rng)

    results = Dict("err"=>out_err_strg, "pred"=>out_pred_strg)
    results = results_synopsis(results, Val(parsed["sweep"]))
    GVFN.save_results(savefile, results, parsed["working"])

end



end
