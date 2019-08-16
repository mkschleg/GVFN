__precompile__(true)

module MackeyGlassExperiment

using GVFN: MackeyGlass, MackeyGlassAgent, step!, start!
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
        default=600000
        "--valSteps"
        help="number of validation steps"
        arg_type=Int64
        default=200000
        "--testSteps"
        help="number of test steps"
        arg_type=Int64
        default=200000
        "--working"
        action=:store_true
    end


    @add_arg_table as begin
        "--horizon"
        help="prediction horizon"
        default=12
        arg_type=Int64
        "--batchsize"
        help="batchsize for models"
        arg_type=Int64
        default=32
        "--alg"
        help="Algorithm"
        default="BatchTD"

        # GVFN
        "--gvfn_opt"
        help="Optimizer"
        default="Descent"
        "--act"
        help="The activation used for the GVFN"
        arg_type=String
        default="linear"
        "--gamma_low"
        arg_type=Float64
        default=0.2
        "--gamma_high"
        arg_type=Float64
        default=0.9
        "--num_gvfs"
        arg_type=Int
        default=128
        "--gvfn_stepsize"
        arg_type=Float64
        default=3e-5

        # Model
        "--model_opt"
        arg_type=String
        default="ADAM"
        "--model_stepsize"
        arg_type=Float64
        default=0.001
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
    num_val = parsed["valSteps"]
    num_test = parsed["testSteps"]
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
        s_tp1 = step!(env)

        if step > horizon
            push!(gt, s_tp1[1])
        end

        pred = step!(agent, s_tp1, 0, false; rng=rng)

        predictions[step] = Flux.data(pred[1])
    end

    valPreds=zeros(Float64,num_val)
    @showprogress 0.1 "Validation Step: " for step in 1:valSteps
        s_tp1= step!(env)
        pred = predict!(agent, s_tp1,0,false;rng=rng)
        valPreds[step] = Flux.data(pred[1])
    end

    testPreds=zeros(Float64,num_test)
    @showprogress 0.1 "Test Step: " for step in 1:testSteps
        s_tp1= step!(env)
        pred = predict!(agent, s_tp1,0,false;rng=rng)
        testPreds[step] = Flux.data(pred[1])
    end

    results = Dict("GroundTruth"=>gt, "Predictions"=>predictions, "ValidationPredictions"=>valPreds,"TestPredictions"=>testPreds)
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

