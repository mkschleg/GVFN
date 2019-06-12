__precompile__(true)

module MackeyGlassExperiment

using GVFN: MackeyGlass, step!, start!
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
        default=60000
        "--working"
        action=:store_true
    end


    # GVFN
    @add_arg_table as begin
        "--max-exponent"
        help="max discount=1.0-2^(-max-exponent)"
        arg_type=Int
        default=7
        "--horizon"
        help="prediction horizon"
        default=12
        arg_type=Int
        "--alg"
        help="Algorithm"
        default="RTD"
        "--params"
        help="Parameters"
        arg_type=Float64
        default=[]
        nargs='+'
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
        "--opt"
        help="Optimizer"
        default="Descent"
        "--optparams"
        help="Parameters"
        arg_type=Float64
        default=[]
        nargs='+'
        "--horde"
        help="The horde used for training"
        default="gamma_chain"
        "--gamma"
        help="The gamma value for the gamma_chain horde"
        arg_type=Float64
        default=0.9
        "--act"
        help="The activation used for the GVFN"
        arg_type=String
        default="sigmoid"
    end

    return as
end


function main_experiment(args::Vector{String})

    as = arg_parse()
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

    horizon = parsed["horizon"]

    predictions = zeros(Float64,num_steps)
    gt = Float64[]

    env = MackeyGlass()
    num_state_features = get_num_features(env)

    s_t = start!(env)

    agent = MackeyGlassAgent(parsed; rng=rng)
    start!(agent, s_t; rng=rng)

    @showprogress 0.1 "Step: " for step in 1:num_steps

        s_tp1 = step!(env, action)

        if step > horizon
            push!(gt, stp1[1])
        end

        pred = step!(agent, s_tp1, 0, false; rng=rng)

        predictions[step] .= Flux.data(pred)
    end

    results = Dict("GroundTruth"=>gt, "Predictions"=>predictions)
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

# CycleWorldExperiment.main_experiment()

