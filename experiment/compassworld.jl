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
    acc = [findmax(results["pred"][step, :])[2] == findmax(results["oracle"][step, :])[2] for step in 1:(size(results["oracle"])[1])]
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[Int64(floor(length(rmse)*0.8)):end]),
        "lc"=>mean(reshape(rmse, 1000, Int64(length(rmse)/1000)); dims=1),
        "acc"=>mean(reshape(acc, 1000, Int64(length(rmse)/1000)); dims=1)
    ])
end

results_synopsis(results, ::Val{false}) = results

function construct_agent(parsed, rng=Random.GLOBAL_RNG)
    out_horde = CWU.get_horde(parsed, "out-")
    ap = CWU.get_behavior_policy(parsed["policy"])

    fc = if "cell" ∈ keys(parsed) && parsed["cell"] ∉ ["ARNN", "ARNNCell"]
        CWU.StandardFeatureCreator()
    else
        CWU.NoActionFeatureCreator()
    end
    fs = feature_size(fc)
    
    τ = parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed["opt"], parsed["alpha"])
    
    initf=(dims...)->glorot_uniform(rng, dims...)
    if "gvfn-horde" ∈ keys(parsed)
        # Construct GVFN Chain
        horde = CWU.get_horde(parsed["gvfn-horde"])
        act = FLU.get_activation(get(parsed, "act", "sigmoid"))
        chain = Flux.Chain(GVFN.GVFR(horde, GVFN.ARNNCell, fs, 3, length(horde), act; init=initf),
                           Flux.data,
                           Dense(length(horde), 32, Flux.relu; initW=initf),
                           Dense(32, length(out_horde); initW=initf))
        GVFN.FluxAgent(out_horde,
                       chain,
                       opt,
                       τ,
                       fc,
                       fs,
                       ap; rng=rng)
    elseif "klength" ∈ keys(parsed)
        rnntype = parsed["cell"]
        if findfirst("Cell", rnntype) == nothing
            rnntype *= "Cell"
        end
        
        forecast_obj = vcat(collect(collect(1:parsed["klength"]) for i in 1:5)...)
        forecast_obj_idx = vcat(collect(fill(i, parsed["klength"]) for i in 1:5)...)

        rnn_func = if rnntype == "ARNNCell"
            GVFN.ARNNCell
        else
            getproperty(Flux, Symbol(rnntype))
        end
        chain = if rnn_func == GVFN.ARNNCell
            Flux.Chain(GVFN.TargetR(rnn_func, fs, 3, length(forecast_obj), init=initf),
                       Flux.data,
                       Dense(length(forecast_obj), 32, Flux.relu; initW=initf),
                       Dense(32, length(out_horde), initW=initf))
        else
            Flux.Chain(GVFN.TargetR(rnn_func, fs, length(forecast_obj), init=initf),
                       Flux.data,
                       Dense(length(forecast_obj), 32, Flux.relu; initW=initf),
                       Dense(32, length(out_horde), initW=initf))
        end

        GVFN.ForecastAgent(out_horde,
                           forecast_obj,
                           forecast_obj_idx,
                           chain,
                           opt,
                           τ, fc, fs, ap;
                           rng=rng)
    else
        # Construct RNN Chain
        rnntype = getproperty(GVFN, Symbol(parsed["cell"]))
        chain = if rnntype == GVFN.ARNN
            Flux.Chain(rnntype(fs, 3, parsed["hidden"]; init=initf),
                       Dense(parsed["hidden"], 32, Flux.relu; initW=initf),
                       Dense(32, length(out_horde); initW=initf))
        else
            Flux.Chain(rnntype(fs, parsed["hidden"]; init=initf),
                       Dense(parsed["hidden"], 32, Flux.relu; initW=initf),
                       Dense(32, length(out_horde); initW=initf))
        end
        GVFN.FluxAgent(out_horde,
                       chain,
                       opt,
                       τ,
                       fc,
                       fs,
                       ap; rng=rng)
    end
    
    

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

function default_arg_dict(agent_type)
    if agent_type == :rnn
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
            "save_dir" => "compassworld_rnn_sgd"
        )
    elseif agent_type == :gvfn
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
            "save_dir" => "compassworld_gvfn_sgd"
        )
    elseif agent_type == :forecast
        Dict{String,Any}(
            "seed" => 4,
            "size" => 8,
            "steps" => 1000000,
            "size" => 6,

            "out-horde" => "forward",
            "policy" => "rafols",
            
            "opt" => "Descent",
            "alpha" => 0.1,
            "truncation" => 1,
            
            "cell" => "RNNCell",            
            "klength" => 8,
            
            "save_dir" => "compassworld_forecast")
    end
end

end
