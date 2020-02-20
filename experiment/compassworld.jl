module CompassWorldExperiment

using GVFN
using Flux
using Flux.Tracker
using Statistics
using Random
using ProgressMeter
using Random

using Reproduce

const CWU = GVFN.CompassWorldUtils
const FLU = GVFN.FluxUtils

function results_synopsis(results, ::Val{true})
    rmse = sqrt.(mean(results["err"].^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[Int64(floor(length(rmse)*0.8)):end]),
        "lc"=>mean(reshape(rmse, 1000, Int64(length(rmse)/1000)); dims=1)
    ])
end

results_synopsis(results, ::Val{false}) = results

function construct_agent(parsed, rng=Random.GLOBAL_RNG)
    out_horde = CWU.get_horde(parsed, "out-")
    ap = CWU.get_behavior_policy(parsed["policy"])

    fc = if "cell" ∈ keys(parsed) && parsed["cell"] != "ARNN"
        CWU.StandardFeatureCreator()
    else
        CWU.NoActionFeatureCreator()
    end
    fs = feature_size(fc)

    initf=(dims...)->glorot_uniform(rng, dims...)
    chain = if "gvfn-horde" ∈ keys(parsed)
        # Construct GVFN Chain
        horde = CWU.get_horde(parsed["gvfn-horde"])
        act = FLU.get_activation(get(parsed, "act", "sigmoid"))
        Flux.Chain(GVFN.GVFR(horde, GVFN.ARNNCell, fs, 3, length(horde), act; init=initf),
                   Flux.data,
                   Dense(length(horde), 32, Flux.relu; initW=initf),
                   Dense(32, length(out_horde); initW=initf))
    else
        # Construct RNN Chain
        rnntype = getproperty(GVFN, Symbol(parsed["cell"]))
        chain = if rnntype == GVFN.ARNN
            Flux.Chain(rnntype(fs, 4, parsed["hidden"]; init=initf),
                       Dense(parsed["hidden"], 32, Flux.relu; initW=initf),
                       Dense(32, length(out_horde); initW=initf))
        else
            Flux.Chain(rnntype(fs, parsed["hidden"]; init=initf),
                       Dense(parsed["hidden"], 32, Flux.relu; initW=initf),
                       Dense(32, length(out_horde); initW=initf))
        end
    end
    
    τ = parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed["opt"], parsed["alpha"])
    
    agent = GVFN.FluxAgent(out_horde,
                           chain,
                           opt,
                           τ,
                           fc,
                           fs,
                           ap; rng=rng)
end

function main_experiment(parsed::Dict; progress=false, working=false)


    num_steps = parsed["steps"]
    seed = parsed["seed"]
    progress = get(parsed, "progress", progress)
    working = get(parsed, "working", working)
    
    savefile = GVFN.save_setup(parsed; working=working)
    if savefile isa Nothing
        return
    end

    rng = Random.MersenneTwister(seed)

    cw_size = get(parsed, "size", 8)
    
    env = CompassWorld(parsed["size"], parsed["size"])
    agent = construct_agent(parsed, rng)

    out_pred_strg = zeros(num_steps, length(agent.horde))
    out_oracle_strg = zeros(num_steps, length(agent.horde))

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    cur_step = 1
    try
        run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
            out_pred_strg[cur_step, :] .= Flux.data(a.out_preds)
            out_oracle_strg[cur_step, :] .= CWU.oracle(env, parsed["out-horde"])
            
            if progress
                ProgressMeter.next!(prg_bar)
            end
            cur_step += 1
    end
        catch exc
        if exc isa ErrorException && (exc.msg == "Loss is infinite" || exc.msg == "Loss is NaN" || exc.msg == "Loss is Inf")
            out_pred_strg[cur_step:end, :] .= Inf
        else
            rethrow()
        end
    end

    sweep = get(parsed, "sweep", false)
    results = Dict("err"=>out_pred_strg .- out_oracle_strg, "pred"=>out_pred_strg, "oracle"=>out_oracle_strg)
    results = results_synopsis(results, Val(sweep))
    GVFN.save_results(savefile, results, working)
end

function default_arg_dict(rnn=false)
    if rnn
        Dict{String, Any}(
            "seed" => 4,
            "size" => 8,

            "steps" => 1000000,
            "out-horde" => "forward",
            "policy" => "rafols",
        
            "opt" => "Descent",
            "alpha" => 0.1,
            "truncation" => 1,
        
            "cell" => "ARNN",
            "hidden" => 45,
            "exp_loc" => "compassworld_rnn_sgd"
        )
    else
        Dict{String, Any}(
            "seed" => 4,
            "size" => 8,

            "steps" => 1000000,
            "out-horde" => "forward",
            "policy" => "rafols",
            
            "opt" => "Descent",
            "alpha" => 0.1,
            "truncation" => 1,
            
            "gvfn-horde"  => "rafols",
            "act" => "sigmoid",
            "exp_loc" => "compassworld_gvfn_sgd"
        )
    end
end

end
