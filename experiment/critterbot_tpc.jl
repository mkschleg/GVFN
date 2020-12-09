module CritterbotTPCExperiment

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

function init_data(num_steps, num_targets, horizon)
    # --- init the buffers we'll save data in ---

    predictions = zeros(Float64, num_steps, num_targets)
    gt = zeros(Float64, num_steps-horizon, num_targets)

    return predictions, gt #, valPreds, vgt, testPreds, tgt
end

function get_env(parsed)
    # --- get the environment ---

    env_t = parsed["env"]

    env = if env_t == "Critterbot"
        if "env_gammas" ∈ keys(parsed)
            CritterbotTPC(parsed["observation_sensors"], parsed["env_gammas"])
        else
            CritterbotTPC(parsed["observation_sensors"])
        end
    else
        throw(DomainError("Environment $(env_t) not implemented, try in Timeseries.jl!"))
    end
    return env
end

function get_agent(parsed, rng)
    # --- get the agent ---

    Agent_t = parsed["agent"]
    if Agent_t == "GVFN"
        agent = GVFN.CritterbotTPCGVFNAgent(parsed; rng=rng)
    elseif Agent_t == "OriginalRNN"
        agent = GVFN.CritterbotTPCOriginalRNNAgent(parsed;rng=rng)
    elseif Agent_t == "RNN"
        agent = GVFN.CritterbotTPCRNNAgent(parsed;rng=rng)
    elseif Agent_t == "AuxTasks"
        agent = GVFN.CritterbotTPCAuxTaskAgent(parsed;rng=rng)
    elseif Agent_t == "OriginalAuxTasks"
        agent = GVFN.CritterbotTPCOriginalAuxTaskAgent(parsed;rng=rng)
    elseif Agent_t == "Tilecoder"
        agent = GVFN.CritterbotTPCTCAgent(parsed; rng=rng)
    elseif Agent_t == "ANN"
        agent = GVFN.CritterbotTPCFCAgent(parsed; rng=rng)
    else
        throw(DomainError("Agent $(Agent_t) not implemented!"))
    end
    return agent
end

# Put the results into a dict with a particular naming convention
label_results(predictions, gt) = Dict("Predictions"=>predictions,
                                      "GroundTruth"=>gt)

default_config(cell="GRU", tau=1, seed=1) = Dict(
    "save_dir"=>"DefaultConfig",
    "exp_file"=>"experiment/timeseries.jl",
    "exp_module_name" => "CritterbotTPCExperiment",
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
    # num_val = parsed["valSteps"]
    # num_test = parsed["testSteps"]

    # seed RNG
    rng = Random.MersenneTwister(seed)

    # get environment
    env = get_env(parsed)
    num_state_features = get_num_features(env)
    num_targets = get_num_targets(env)

    # init data buffers
    # println(num_steps, " ", num_targets, " ", horizon)
    predictions, gt = init_data(num_steps, num_targets, horizon)

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

            s_tp1, r = step!(env)
            
            pred = step!(agent, s_tp1, Float32.(r), false, rng)
            
            # println(size(pred))
            predictions[step, :] .= Flux.data(pred)

            if step > horizon
                gt[step-horizon, :] = GVFN.ground_truth(env)
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

    # put the arrays in a dict
    results = label_results(predictions, gt) #, valPreds, vgt, testPreds, tgt)

    # save results
    GVFN.save_results(savefile, results, working)

    return results
end

end
