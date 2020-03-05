module TimeSeriesExperiment

using GVFN: MackeyGlass, MSO, ACEA, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using Flux.Tracker: TrackedArray, TrackedReal, track, @grad
using DataStructures: CircularBuffer

# =================
# --- UTILITIES ---
# =================

function init_data(num_steps, num_val, num_test, horizon)
    # --- init the buffers we'll save data in ---

    predictions = zeros(Float64,num_steps)
    gt = zeros(Float64, num_steps-horizon)

    valPreds=zeros(Float64,num_val)
    vgt = zeros(Float64, num_val - horizon)

    testPreds=zeros(Float64,num_test)
    tgt = zeros(Float64, num_test-horizon)

    return predictions, gt, valPreds, vgt, testPreds, tgt
end

function get_env(parsed)
    # --- get the environment ---

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
    return env
end

function get_agent(parsed, rng)
    # --- get the agent ---

    Agent_t = parsed["agent"]
    if Agent_t == "GVFN"
        agent = GVFN.TimeSeriesGVFNAgent(parsed; rng=rng)
    elseif Agent_t == "RNN"
        agent = GVFN.TimeSeriesRNNAgent(parsed;rng=rng)
    else
        throw(DomainError("Agent $(Agent_t) not implemented!"))
    end
    return agent
end

# Put the results into a dict with a particular naming convention
label_results(predictions, gt, valPreds, vgt, testPreds, tgt) = Dict("Predictions"=>predictions,
                                                                     "GroundTruth"=>gt,
                                                                     "ValidationPredictions"=>valPreds,
                                                                     "ValidationGroundTruth"=>vgt,
                                                                     "TestPredictions"=>testPreds,
                                                                     "TestGroundTruth"=>tgt)

# ==================
# --- EXPERIMENT ---
# ==================


function main_experiment(parsed::Dict; working = false, progress=false)

    savefile = GVFN.save_setup(parsed; save_dir_key="save_dir", working=working)
    if savefile == nothing
        return
    end

    # get parsed args
    seed = parsed["seed"]

    horizon = parsed["horizon"]
    num_steps = parsed["steps"]
    num_val = parsed["valSteps"]
    num_test = parsed["testSteps"]

    # seed RNG
    rng = Random.MersenneTwister(seed)

    # init data buffers
    predictions, gt, valPreds, vgt, testPreds, tgt = init_data(num_steps, num_val, num_test, horizon)

    # get environment
    env = get_env(parsed)
    num_state_features = get_num_features(env)

    # get agent
    agent = get_agent(parsed, rng)

    # start experiment
    s_t = start!(env)
    start!(agent, s_t, rng)

    # Go through the training phase. Catch errors related to infinite loss, saving dummy data
    # and quitting early
    try

        # progress bar
        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

        for step in 1:num_steps
            s_tp1 = step!(env)

            pred = step!(agent, s_tp1, 0, false, rng)
            predictions[step] = pred[1]

            if step > horizon
                gt[step-horizon] = s_tp1[1]
            end

            if progress
                ProgressMeter.next!(prg_bar)
            end
        end

    catch exc
        # Save dummy data and quit early
        if exc isa ErrorException && (exc.msg == "Loss is infinite" || exc.msg == "Loss is NaN" || exc.msg == "Loss is Inf")
            predictions .= Inf
            valPreds .= Inf
            testPreds .= Inf

            results = label_results(predictions, gt, valPreds, vgt, testPreds, tgt)
            GVFN.save_results(savefile, results, working)
        else
            rethrow()
        end
        return results
    end

    # Re-init progress bar for validation phase
    prg_bar = ProgressMeter.Progress(num_val, "Validation Step: ")

    # predict on the validation data
    for step in 1:num_val
        s_tp1= step!(env)

        pred = predict!(agent, s_tp1,0,false, rng)
        valPreds[step] = Flux.data(pred[1])

        if step>horizon
            vgt[step-horizon] = s_tp1[1]
        end

        if progress
            ProgressMeter.next!(prg_bar)
        end
    end

    # re-init progress bar for test phase
    prg_bar = ProgressMeter.Progress(num_test, "Test Step: ")

    # predict on the test data
    for step in 1:num_test
        s_tp1= step!(env)

        pred = predict!(agent, s_tp1, 0, false, rng)
        testPreds[step] = Flux.data(pred[1])

        if step>horizon
            tgt[step - horizon] = s_tp1[1]
        end

        if progress
            ProgressMeter.next!(prg_bar)
        end
    end

    # put the arrays in a dict
    results = label_results(predictions, gt, valPreds, vgt, testPreds, tgt)

    # save results
    GVFN.save_results(savefile, results, working)

    return results
end

end
