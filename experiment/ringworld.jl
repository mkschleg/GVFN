module RingWorldExperiment

using GVFN
using Flux
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const RWU = GVFN.RingWorldUtils
const FLU = GVFN.FluxUtils

function results_synopsis(results, ::Val{true})
    rmse = sqrt.(mean(results["err"].^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[Int(floor(length(rmse)*0.8)):end]),
        "lc"=>mean(reshape(rmse, 1000, :); dims=1)[1,:]
    ])
end
results_synopsis(results, ::Val{false}) = results

function construct_agent(parsed, rng=Random.GLOBAL_RNG)

    out_horde = RWU.get_horde(parsed["outhorde"], parsed["size"], get(parsed, "outgamma", 0.9f0))
    ap = GVFN.RandomActingPolicy([0.5f0, 0.5f0])
    
    # GVFN horde
    fc = if "cell" ∈ keys(parsed) && parsed["cell"] != "ARNN"
        RWU.StandardFeatureCreatorWithAction()
    else
        RWU.StandardFeatureCreator()    
    end
    fs = feature_size(fc)

    initf=(dims...)->glorot_uniform(rng, dims...)

    chain = if "horde" ∈ keys(parsed)
        horde = RWU.get_horde(parsed)
        chain = Flux.Chain(GVFN.GVFR(horde, GVFN.ARNNCell, fs, 3, length(horde), Flux.sigmoid; init=initf),
                           Flux.data,
                           Dense(length(horde), 16, Flux.relu; initW=initf),
                           Dense(16, length(out_horde); initW=initf))
    else
        rnntype = getproperty(GVFN, Symbol(parsed["cell"]))
        chain = if rnntype == GVFN.ARNN
            Flux.Chain(rnntype(fs, 4, parsed["hidden"]; init=initf),
                       Dense(parsed["hidden"], 16, Flux.relu; initW=initf),
                       Dense(16, length(out_horde); initW=initf))
        else
            Flux.Chain(rnntype(fs, parsed["hidden"]),
                       Dense(parsed["hidden"], 16, Flux.relu; initW=initf),
                       Dense(16, length(out_horde); initW=initf))
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



function main_experiment(parsed::Dict; working=false, progress=false)

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    progress = get(parsed, "progress", progress)
    working = get(parsed, "working", working)
    
    savefile = GVFN.save_setup(parsed; save_dir_key="save_dir", working=working)
    if savefile isa Nothing
        return
    end

    rng = Random.MersenneTwister(seed)

    # Construct Environment
    env = RingWorld(parsed["size"])
    agent = construct_agent(parsed, rng)

    out_pred_strg = zeros(num_steps, length(agent.horde))
    out_oracle_strg = zeros(num_steps, length(agent.horde))

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    cur_step = 1
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        out_pred_strg[cur_step, :] .= Flux.data(a.out_preds)
        out_oracle_strg[cur_step, :] .= RWU.oracle(env, parsed["outhorde"], get(parsed, "outgamma", 0.9f0))

        if progress
           ProgressMeter.next!(prg_bar)
        end
        cur_step += 1
    end

    sweep = get(parsed, "sweep", false)    
    results = Dict("err"=>out_pred_strg .- out_oracle_strg, "pred"=>out_pred_strg, "oracle"=>out_oracle_strg)
    results = results_synopsis(results, Val(sweep))
    GVFN.save_results(savefile, results, working)
end


function default_arg_dict(rnn=false)
    if rnn
        Dict{String,Any}(
            "seed" => 2,
            "size" => 6,
            "steps" => 300000,
            
            "outhorde" => "gammas_term",
            "outgamma" => 0.9,
            
            "opt" => "Descent",
            "alpha" => 0.1,
            "truncation" => 2,

            "cell"=>"ARNN",
            "hidden"=>14,

            "save_dir" => "ringworld_rnn")
    else
        Dict{String,Any}(
            "seed" => 2,
            "size" => 6,
            "steps" => 300000,

            "outhorde" => "gammas_term",
            "outgamma" => 0.9,
            
            "opt" => "Descent",
            "alpha" => 0.1,
            "truncation" => 2,

            "act" => "sigmoid",
            "horde" => "gamma_chain",
            "gamma" => 0.95,

            "save_dir" => "ringworld_gvfn")
    end
end

end
