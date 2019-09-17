__precompile__(true)

module CompassWorldRNNActionExperiment

using GVFN: CycleWorld, step!, start!
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
# using Reproduce
using Random

using Flux.Tracker: TrackedArray, TrackedReal, track, @grad

using DataStructures: CircularBuffer

const cwu = GVFN.CompassWorldUtils

function arg_parse(as::ArgParseSettings = ArgParseSettings())

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
        "--sweep"
        action=:store_true
        "--verbose"
        action=:store_true
        "--working"
        action=:store_true
        "--progress"
        action=:store_true
    end


    #Compass world settings
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

    # shared settings
    @add_arg_table as begin
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
        "--horde"
        help="The horde used for training"
        default="gamma_chain"
    end

    # RNN Settings
    @add_arg_table as begin
        "--opt"
        help="Optimizer"
        default="Descent"
        "--optparams"
        help="Parameters"
        arg_type=Float64
        default=[]
        nargs='+'
        "--cell"
        help="Cell"
        default="RNNCell"
        "--numhidden"
        help="Number of hidden units in cell"
        arg_type=Int64
        "--feature"
        help="The feature creator to use"
        arg_type=String
        default="standard"
    end

    return as
end

function results_synopsis(err, ::Val{true})
    rmse = sqrt.(mean(err.^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[Int64(floor(length(rmse)*0.8)):end]),
        "lc"=>reshape(rmse, 1000, Int64(length(rmse)/1000))
    ])
end

results_synopsis(err, ::Val{false}) = sqrt.(mean(err.^2; dims=2))

function main_experiment(args::Vector{String})

    #####
    # Setup experiment environment
    #####
    as = arg_parse()
    parsed = parse_args(args, as)

    savepath = ""
    savefile = ""
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, "results.jld2")
        if isfile(savefile)
            println("File exists")
            return
        end
    end

    ####
    # General Experiment parameters
    ####
    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = CompassWorld(parsed["size"], parsed["size"])
    num_state_features = get_num_features(env)


    _, s_t = start!(env)

    out_horde = cwu.forward()

    out_pred_strg = zeros(num_steps, length(out_horde))
    out_err_strg = zeros(num_steps, length(out_horde))

    fc = cwu.StandardFeatureCreator()
    if parsed["feature"] == "action"
        fc = cwu.ActionTileFeatureCreator()
    end

    fs = JuliaRL.FeatureCreators.feature_size(fc)

    # ap = cwu.ActingPolicy()
    ap = cwu.get_behavior_policy(parsed["policy"])
    
    # agent = RNNAgent(parsed; rng=rng)
    agent = GVFN.RNNActionAgent(out_horde, fc, fs,
                                3, ap, parsed;
                                rng=rng,
                                init_func=(dims...)->glorot_uniform(rng, dims...))
    action = start!(agent, s_t; rng=rng)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    progress = parsed["progress"]
    
    for step in 1:num_steps
    # for step in 1:num_steps
        if step%100000 == 0
            # println("Garbage Clean!")
            GC.gc()
        end
        if parsed["verbose"]
            if step%10000 == 0
                print(step, "\r")
            end
        end

        _, s_tp1, _, _ = step!(env, action)
        out_preds, action = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step, :] .= Flux.data.(out_preds)
        out_err_strg[step, :] .= out_pred_strg[step, :] .- cwu.oracle(env, parsed["horde"])

        if progress
           next!(prg_bar)
        end
    end

    results = results_synopsis(out_err_strg, Val(parsed["sweep"]))
    if !parsed["working"]
        JLD2.@save savefile results
    else
        return out_pred_strg, out_err_strg, results
    end
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

# CycleWorldExperiment.main_experiment()


