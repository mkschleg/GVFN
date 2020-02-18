module CycleWorldExperiment

using Flux
using GVFN
# using Statistics
using Random
using ProgressMeter
using Random

const CWU = GVFN.CycleWorldUtils
const FLU = GVFN.FluxUtils

function construct_agent(parsed, rng=RNG.GLOBAL_RNG)

    out_horde = CWU.onestep(parsed["chain"])
    fc = (state, action)->CWU.build_features_cycleworld(state)
    fs = 2
    ap = GVFN.RandomActingPolicy([1.0])
    initf=(dims...)->glorot_uniform(rng, dims...)

    chain = if "horde" ∈ keys(parsed)
        # GVFN Initialization
        horde = CWU.get_horde(parsed)
        act = FLU.get_activation(get(parsed, "act", "sigmoid")) # activation w/ default as sigmoid
        Flux.Chain(GVFN.GVFR(horde, Flux.RNNCell, fs, length(horde), act, init=initf),
                   Flux.data,
                   Dense(length(horde), length(out_horde), initW=initf))
    else
        # RNN Initialization
        rnntype = getproperty(Flux, Symbol(parsed["cell"]))
        Flux.Chain(rnntype(fs, parsed["hidden"], init=initf),
                   Dense(parsed["hidden"], length(out_horde), initW=initf))
    end

    τ = parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed["opt"], parsed["alpha"])

    GVFN.FluxAgent(out_horde,
                   chain,
                   opt,
                   τ, fc, fs, ap;
                   rng=rng)
end

function main_experiment(parsed::Dict; progress=false, working=false)

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    progress = get(parsed, "progress", progress)
    working = get(parsed, "working", working)

    savefile = GVFN.save_setup(parsed; save_dir_key="save_dir", working=working)

    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])
    agent = construct_agent(parsed, rng)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    out_pred_strg = zeros(num_steps)
    out_err_strg = zeros(num_steps)
    cur_step = 1
    
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        out_pred_strg[cur_step] = Flux.data(a.out_preds)[1]
        out_err_strg[cur_step] = out_pred_strg[cur_step][1] - CWU.oracle(env, "onestep", 0.0)[1]

        if progress
           ProgressMeter.next!(prg_bar)
        end
        cur_step += 1
    end
    
    results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg])
    GVFN.save_results(savefile, results, working)
end


function default_arg_dict(rnn=false)
    if rnn
        Dict{String,Any}(
            "seed" => 1,
            "steps" => 100000,
            "chain" => 6,
            "opt" => "Descent",
            "alpha" => 0.1,
            "truncation" => 4,

            "cell" => "RNN",
            "numhidden" => 7,

            "exp_loc" => "cycleworld_rnn_sweep_sgd_lin")
    else
        Dict{String,Any}(
            "seed" => 1,
            "steps" => 100000,
            "chain" => 6,
            "opt" => "Descent",
            "alpha" => 0.1,
            "truncation" => 4,
            
            "act" => "sigmoid",
            "horde" => "gamma_chain",
            "gamma" => 0.9,

            "exp_loc" => "cycleworld_gvfn_sweep_sgd_lin")
    end
end

end
