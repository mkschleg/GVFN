__precompile__(true)

module CycleWorldExperiment

import Flux
import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

# using GVFN: CycleWorld, step!, start!
using GVFN: CycleWorld, step!, start!
using GVFN: CycleWorldAgent
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using DataStructures: CircularBuffer


# include("utils/util.jl")
import GVFN.CycleWorldUtils

# function Flux.Optimise.apply!(o::Flux.RMSProp, x, Δ)
#   η, ρ = o.eta, o.rho
#   acc = get!(o.acc, x, zero(x))::typeof(Flux.data(x))
#   @. acc = ρ * acc + (1 - ρ) * Δ^2
#   @. Δ *= η / (√acc + Flux.Optimise.ϵ)
# end

function arg_parse(as::ArgParseSettings = ArgParseSettings())

    #Experiment
    @add_arg_table as begin
        "--exp_loc"
        help="Location of experiment"
        arg_type=String
        default="tmp"
        "--seed"
        help="Seed of rng"
        arg_type=Int64
        default=0
        "--steps"
        help="number of steps"
        arg_type=Int64
        default=100
        "--verbose"
        action=:store_true
        "--working"
        action=:store_true
    end

    #Cycle world
    @add_arg_table as begin
        "--chain"
        help="The length of the cycle world chain"
        arg_type=Int64
        default=6
    end

    # shared settings
    @add_arg_table as begin
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
    end

    # GVFN 
    @add_arg_table as begin
        "--alg"
        help="Algorithm"
        default="TDLambda"
        "--params"
        help="Parameters"
        arg_type=Float64
        nargs='+'
        "--opt"
        help="Optimizer"
        default="Descent"
        "--optparams"
        help="Parameters"
        arg_type=Float64
        default=[]
        nargs='+'
        "--horde"
        help="The horde used for training"
        default="gamma_chain"
        "--gamma"
        help="The gamma value for the gamma_chain horde"
        arg_type=Float64
        default=0.9
        "--act"
        help="Activation function for GVFN"
        arg_type=String
        default="identity"
    end
    return as
end

function oracle(env::CycleWorld, horde_str, γ=0.9)
    chain_length = env.chain_length
    state = env.agent_state
    ret = Array{Float64,1}()
    if horde_str == "chain"
        ret = zeros(chain_length)
        ret[chain_length - state] = 1
    elseif horde_str == "gamma_chain"
        ret = zeros(chain_length + 1)
        ret[chain_length - state] = 1
        ret[end] = γ^(chain_length - state - 1)
    elseif horde_str == "gammas"
        ret = collect(0.0:0.1:0.9).^(chain_length - state - 1)
    elseif horde_str == "onestep"
        ret = zeros(chain_length)
        ret = chain_length - state == 1 ? 1.0 : 0.0
    else
        throw("Bug Found")
    end

    return ret
end

function main_experiment(args::Vector{String})

    as = arg_parse()
    parsed = parse_args(args, as)

    savepath = ""
    savefile = ""
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, "results.jld2")
        if isfile(savefile)
            return
        end
    end

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])

    out_pred_strg = zeros(num_steps)
    out_err_strg = zeros(num_steps)

    _, s_t = start!(env)

    agent = CycleWorldAgent(parsed; rng=rng)
    start!(agent, s_t; rng=rng)

    for step in 1:num_steps

        _, s_tp1, _, _ = step!(env, 1)
        out_preds, action = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step] = Flux.data(out_preds)[1]
        out_err_strg[step] = out_pred_strg[step][1] - oracle(env, "onestep", parsed["gamma"])[1]
    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])

    if !parsed["working"]
        JLD2.@save savefile results
    else
        return results
    end

end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

