__precompile__(true)

module TimeSeriesExperiment

using GVFN: MackeyGlass, MSO, ACEA, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
using Random
using ProgressMeter
using JLD2
using Reproduce
using Reproduce.Config
using Random
using Flux.Tracker: TrackedArray, TrackedReal, track, @grad

using DataStructures: CircularBuffer

function main_experiment(cfg::ConfigManager, save_loc::String=string(@__DIR__))

    parsed = cfg["args"]
    seed = cfg["run"]+203857

    num_steps = parsed["steps"]
    num_val = parsed["valSteps"]
    num_test = parsed["testSteps"]
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
        agent = GVFN.TimeSeriesGVFNAgent(parsed; rng=rng)
    elseif Agent_t == "RNN"
        agent = GVFN.TimeSeriesRNNAgent(parsed;rng=rng)
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
    save(cfg, results)
    return results
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end
