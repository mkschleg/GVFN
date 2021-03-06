module CycleWorldForecastExperiment

using Flux
using GVFN
# using Statistics
using Random
using ProgressMeter
using Random

const CWU = GVFN.CycleWorldUtils
const FLU = GVFN.FluxUtils

function default_arg_dict()

    Dict{String,Any}(
        "seed" => 1,
        "steps" => 100000,
        "chain" => 6,
        "opt" => "Descent",
        "alpha" => 0.1,
        "truncation" => 1,
        "klength" => 6,
        "save_dir" => "cycleworld_forecast")

end


function construct_agent(parsed, rng=RNG.GLOBAL_RNG)

    out_horde = CWU.onestep(parsed["chain"])
    fc = (state, action)->CWU.build_features_cycleworld(state)
    fs = 2
    ap = GVFN.RandomActingPolicy([1.0])
    forecast_obj = collect(1:parsed["klength"])
    forecast_obj_idx = fill(2, length(forecast_obj))

    
    initf=(dims...)->glorot_uniform(rng, dims...)
    
    rnntype = get(parsed, "cell", "RNNCell")
    if findfirst("Cell", rnntype) == nothing
        rnntype *= "Cell"
    end
    
    rnn_func = getproperty(Flux, Symbol(rnntype))
    chain = Flux.Chain(GVFN.TargetR(rnn_func, fs, length(forecast_obj), init=initf),
                       Flux.data,
                       Dense(length(forecast_obj), length(out_horde), initW=initf))

    τ = parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed["opt"], parsed["alpha"])

    GVFN.ForecastAgent(out_horde,
                       forecast_obj,
                       forecast_obj_idx,
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




end
