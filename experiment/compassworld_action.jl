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

using Flux.Tracker: TrackedArray, TrackedReal, track, @grad

using DataStructures: CircularBuffer

const cwu = GVFN.CompassWorldUtils

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


function results_synopsis(err, ::Val{true})
    rmse = sqrt.(mean(err.^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[Int64(floor(length(rmse)*0.8)):end]),
        "lc"=>mean(reshape(rmse, 1000, Int64(length(rmse)/1000));dims=1)'
    ])
end

results_synopsis(err, ::Val{false}) = sqrt.(mean(err.^2; dims=2))


function main_experiment(args::Vector{String})

    as = arg_parse()
    parsed = parse_args(args, as)

    ######
    # Experiment Setup
    ######
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

    out_pred_strg = zeros(num_steps, 5)
    out_err_strg = zeros(num_steps, 5)

    ######
    # Environment Setup
    ######
    
    env = CompassWorld(parsed["size"], parsed["size"])
    num_state_features = get_num_features(env)

    _, s_t = start!(env) # Start environment

    #####
    # Agent specific setup.
    #####
    
    horde = cwu.get_horde(parsed)
    out_horde = cwu.get_horde(parsed, "out")

    fc = cwu.StandardFeatureCreator()
    if parsed["feature"] == "action"
        fc = cwu.ActionTileFeatureCreator()
    end

    fs = JuliaRL.FeatureCreators.feature_size(fc)

    ap = cwu.get_behavior_policy(parsed["policy"])
    
    agent = GVFN.GVFNActionAgent(horde, out_horde,
                                 fc, fs,
                                 3,
                                 ap,
                                 parsed;
                                 rng=rng,
                                 init_func=(dims...)->glorot_uniform(rng, dims...))
    
    action = start!(agent, s_t; rng=rng) # Start agent
    verbose = parsed["verbose"]
    progress = parsed["progress"]

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    for step in 1:num_steps

        _, s_tp1, _, _ = step!(env, action)
        out_preds, action = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step, :] .= Flux.data(out_preds)
        out_err_strg[step, :] .= out_pred_strg[step, :] .- cwu.oracle(env, parsed["outhorde"])

        if verbose
            println("step: $(step)")
            println(env)
            println(agent)
            println(out_preds)
        end

        if progress
           next!(prg_bar)
        end

        
    end

    results = results_synopsis(out_err_strg, Val(parsed["sweep"]))
    
    if !parsed["working"]
        JLD2.@save savefile results
    else
        return results, out_pred_strg, out_err_strg
    end
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

# CycleWorldExperiment.main_experiment()


