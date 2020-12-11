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

function init_data(num_steps, num_targets, num_val, num_test, horizon)
    # --- init the buffers we'll save data in ---

    predictions = zeros(Float64, num_steps, num_targets)
    gt = zeros(Float64, num_steps-horizon, num_targets)

    valPreds=zeros(Float64,num_val, num_targets)
    vgt = zeros(Float64, num_val - horizon, num_targets)

    testPreds=zeros(Float64,num_test, num_targets)
    tgt = zeros(Float64, num_test-horizon, num_targets)

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
    elseif env_t == "Critterbot"
        env = Critterbot(parsed["observation_sensors"], parsed["target_sensors"])
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
    elseif Agent_t == "GVFNJoint"
        agent = GVFN.TimeSeriesGVFNJointAgent(parsed; rng=rng)
    elseif Agent_t == "OriginalRNN"
        agent = GVFN.TimeSeriesOriginalRNNAgent(parsed;rng=rng)
    elseif Agent_t == "RNN"
        agent = GVFN.TimeSeriesRNNAgent(parsed;rng=rng)
    elseif Agent_t == "AuxTasks"
        agent = GVFN.TimeSeriesAuxTaskAgent(parsed;rng=rng)
    elseif Agent_t == "OriginalAuxTasks"
        agent = GVFN.TimeSeriesOriginalAuxTaskAgent(parsed;rng=rng)
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

default_config(cell="GRU", tau=1, seed=1) = Dict(
    "save_dir"=>"DefaultConfig",
    "exp_file"=>"experiment/timeseries.jl",
    "exp_module_name" => "TimeSeriesExperiment",
    "exp_func_name" => "main_experiment",
    "arg_iter_type" => "iter",


    "env" => "MackeyGlass",
    "horizon" => 12,
    "steps" => 600000,
    "valSteps" => 200000,
    "testSteps" => 200000,

    #"agent" => "GVFN",
    "agent" => "RNN",
    #"agent" => "AuxTasks",
    "activation" => "relu",
    "update_fn" => "BatchTD",
    "batchsize" => 32,

    "horde" => "LinSpacing",
    "gamma_low" => 0.1,
    "gamma_high" => 0.97,
    "num_gvfs" => 128,

    # "gvfn_tau" => 1,
    # "gvfn_stepsize" => 3e-5,
    # "gvfn_opt" => "Descent",
    # "model_opt" => "ADAM",
    # "model_stepsize" => 0.001,

    "rnn_opt" => "ADAM",
    "rnn_lr" => 0.001,
    "rnn_beta1" => 0.99,
    "rnn_beta2" => 0.999,

    "rnn_cell" => cell,
    "rnn_nhidden" => 128,
    "rnn_activation" => "tanh",
    "rnn_tau" => tau,

    "model_clip_coeff"=>0.25,

    "seed" => seed,
)

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

    # get environment
    env = get_env(parsed)
    num_state_features = get_num_features(env)
    num_targets = get_num_targets(env)

    # init data buffers
    predictions, gt, valPreds, vgt, testPreds, tgt = init_data(num_steps, num_targets, num_val, num_test, horizon)

    # get agent
    parsed["num_features"] = num_state_features
    parsed["num_targets"] = get_num_targets(env)
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

            predictions[step, :] .= Flux.data(pred)

            if step > horizon
                gt[step-horizon, :] = s_tp1
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
        valPreds[step, :] .= Flux.data(pred)


        if step>horizon
            vgt[step-horizon,:] = s_tp1
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
        testPreds[step,:] .= Flux.data(pred)

        if step>horizon
            tgt[step - horizon,:] = s_tp1
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
