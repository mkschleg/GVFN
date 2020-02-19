__precompile__(true)

module TimeSeriesExperiment

using GVFN: MackeyGlass, MSO, ACEA, TimeSeriesAgent, step!, start!
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
        "--env"
        help="Name of the time series dataset to use"
        arg_type=String
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
        "--agent"
        help="which agent to use"
        arg_type=String
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

        "--normalizer"
        help="input normalizer"
        arg_type=String
        default="Identity"
        "--max"
        help="max normalizing constant"
        arg_type=Float64
        default=1.0
        "--min"
        help="min normalizing constant"
        arg_type=Float64
        default=0.0

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

        # RNN
        # --horizon
        # --batchsize
        "--rnn_opt"
        help="Optimizer"
        default="Adam"

        "--rnn_tau"
        help="BPTT truncation length"
        arg_type=Int
        default=1

        "--rnn_lr"
        help="learning rate"
        arg_type=Float64
        default=0.0001

        "--rnn_nhidden"
        help="number of hidden units"
        arg_type=Int
        default=64

        "--rnn_cell"
        help="RNN cell to use (e.g. GRU)"
        arg_type=String
        default="GRU"
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
    gt = zeros(Float64, num_steps-horizon)


    env_t = parsed["env"]
    if env_t == "MackeyGlass"
        env = MackeyGlass()
    elseif env_t == "MSO"
        env = MSO()
    elseif env_t == "ACEA"
        env = ACEA()
    else
        throw(DomainError("Environment $(env_t) not implemented!"))
    end
    num_state_features = get_num_features(env)

    s_t = start!(env)

    Agent_t = parsed["agent"]
    if Agent_t == "GVFN"
        agent = GVFN.TimeSeriesFluxAgent(parsed; rng=rng)
    elseif Agent_t == "RNN"
        agent = TimeSeriesRNNAgent(parsed;rng=rng)
    else
        throw(DomainError("Agent $(Agent_t) not implemented!"))
    end

    start!(agent, s_t, rng)

    @showprogress 0.1 "Step: " for step in 1:num_steps
        s_tp1 = step!(env)

        pred = step!(agent, s_tp1, 0, false, rng)
        predictions[step] = pred[1]

        if step > horizon
            gt[step-horizon] = s_tp1[1]
        end
    end

    valPreds=zeros(Float64,num_val)
    vgt = zeros(Float64, num_val - horizon)
    @showprogress 0.1 "Validation Step: " for step in 1:num_val
        s_tp1= step!(env)

        pred = predict!(agent, s_tp1,0,false, rng)
        valPreds[step] = Flux.data(pred[1])

        if step>horizon
            vgt[step-horizon] = s_tp1[1]
        end
    end

    testPreds=zeros(Float64,num_test)
    tgt = zeros(Float64, num_test-horizon)
    @showprogress 0.1 "Test Step: " for step in 1:num_test
        s_tp1= step!(env)

        pred = predict!(agent, s_tp1, 0, false, rng)
        testPreds[step] = Flux.data(pred[1])

        if step>horizon
            tgt[step - horizon] = s_tp1[1]
        end
    end

    results = Dict("GroundTruth"=>gt,
                   "Predictions"=>predictions,
                   "ValidationPredictions"=>valPreds,
                   "TestPredictions"=>testPreds,
                   "TestGroundTruth"=>tgt,
                   "ValidationGroundTruth"=>vgt)
    JLD2.@save savefile results
    return results
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end
