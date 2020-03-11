
module RingWorldAuxTaskExperiment

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
    at_horde = RWU.get_horde(parsed["at-horde"], parsed["size"], get(parsed, "at-horde", 0.9f0), length(out_horde))

    horde = GVFN.merge(out_horde, at_horde)
    
    ap = GVFN.RandomActingPolicy([0.5f0, 0.5f0])
    
    # GVFN horde
    fc = if parsed["cell"] ∉ ["ARNN", "ARNNCell"]
        RWU.StandardFeatureCreatorWithAction()
    else
        RWU.StandardFeatureCreator()    
    end
    fs = feature_size(fc)

    τ = parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed["opt"], parsed["alpha"])

    initf=(dims...)->glorot_uniform(rng, dims...)
    rnntype = getproperty(GVFN, Symbol(parsed["cell"]))
    chain = if rnntype == GVFN.ARNN
        Flux.Chain(rnntype(fs, 2, parsed["hidden"]; init=initf),
                   Dense(parsed["hidden"], 16, Flux.relu; initW=initf),
                   Dense(16, length(horde); initW=initf))
    else
        Flux.Chain(rnntype(fs, parsed["hidden"]),
                   Dense(parsed["hidden"], 16, Flux.relu; initW=initf),
                   Dense(16, length(horde); initW=initf))
    end

    agent = GVFN.FluxAgent(out_horde,
                           chain,
                           opt,
                           τ,
                           fc,
                           fs,
                           ap; rng=rng)
    agent, out_horde, at_horde
end

function main_experiment(parsed::Dict; working=false, progress=false)

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    progress = get(parsed, "progress", progress)
    working = get(parsed, "working", working)

    # default arguments
    num_hidden = get!(parsed, "hidden", parsed["size"]*2 + 2)
    
    savefile = GVFN.save_setup(parsed; save_dir_key="save_dir", working=working)
    if savefile isa Nothing
        return
    end

    rng = Random.MersenneTwister(seed)

    # Construct Environment
    env = RingWorld(parsed["size"])
    agent, out_horde, at_horde = construct_agent(parsed, rng)

    eval_pred_strg = zeros(num_steps, length(out_horde))
    eval_oracle_strg = zeros(num_steps, length(out_horde))

    at_pred_strg = zeros(num_steps, length(at_horde))
    at_oracle_strg = zeros(num_steps, length(at_horde))


    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    cur_step = 1
    try
        run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
            eval_pred_strg[cur_step, :] .= Flux.data(a.out_preds)[1:length(out_horde)]
            eval_oracle_strg[cur_step, :] .= RWU.oracle(env, parsed["outhorde"], get(parsed, "outgamma", 0.9f0))

            at_pred_strg[cur_step, :] .= Flux.data(a.out_preds)[length(out_horde)+1:end]
            at_oracle_strg[cur_step, :] .= RWU.oracle(env, parsed["at-horde"], get(parsed, "outgamma", 0.9f0))

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
    Dict{String,Any}(
        "seed" => 2,
        "size" => 6,
        "steps" => 300000,
        
        "outhorde" => "onestep",
        "outgamma" => 0.9,

        "outhorde" => "chain",
        "outgamma" => 0.9,
        
        "opt" => "Descent",
        "alpha" => 0.1,
        "truncation" => 2,
        
        "cell"=>"ARNN",
        "hidden"=>14,

        "save_dir" => "ringworld_rnn")
end

