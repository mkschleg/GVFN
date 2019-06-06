__precompile__(true)

module CycleWorldRNNExperiment

using GVFN: CycleWorld, step!, start!
using GVFN
import Flux
import Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter
using FileIO
using ArgParse
using Random
using DataStructures: CircularBuffer

# include("utils/util.jl")
import GVFN.CycleWorldUtils
import GVFN.FluxUtils

function Flux.Optimise.apply!(o::Flux.RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(Flux.data(x))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + Flux.Optimise.ϵ)
end

function exp_settings(as::ArgParseSettings = ArgParseSettings())

    #Experiment
    @add_arg_table as begin
        "--seed"
        help="Seed of rng"
        arg_type=Int64
        default=0
        "--steps"
        help="number of steps"
        arg_type=Int64
        default=100
        "--savefile"
        help="save file for experiment"
        arg_type=String
        default="temp.jld"
        "--verbose"
        action=:store_true
        "--working"
        action=:store_true
    end

    #Cycle world specific settings
    CycleWorldUtils.env_settings!(as)
    CycleWorldUtils.horde_settings!(as)

    # RNN
    FluxUtils.rnn_settings!(as)
    FluxUtils.opt_settings!(as)

    return as
end

build_features(s) = [1.0, s[1], 1-s[1]]

function main_experiment(args::Vector{String})

    as = exp_settings()
    parsed = parse_args(args, as)

    savefile = parsed["savefile"]
    savepath = dirname(savefile)

    if savepath != ""
        if !isdir(savepath)
            mkpath(savepath)
        end
    end

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])

    horde = CycleWorldUtils.get_horde(parsed)

    num_gvfs = length(horde)

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)
    rnn = FluxUtils.construct_rnn(3, parsed)
    out_model = Flux.Dense(parsed["numhidden"], length(horde))

    out_pred_strg = zeros(num_steps, num_gvfs)
    out_err_strg = zeros(num_steps, num_gvfs)

    _, s_t = start!(env)

    state_list = CircularBuffer{Array{Float32, 1}}(τ+1)
    fill!(state_list, zeros(3))
    push!(state_list, build_features(s_t))
    hidden_state_init = GVFN.get_initial_hidden_state(rnn)

    lu = OnlineTD_RNN(state_list, hidden_state_init)

    for step in 1:num_steps
        if parsed["verbose"]
            if step % 1000 == 0
                print(step, "\r")
            end
        end
        _, s_tp1, _, _ = step!(env, 1)

        preds = train_step!(out_model, rnn, horde, opt, lu, build_features(s_tp1), s_tp1)

        out_pred_strg[step,:] = Flux.data(preds[end])
        out_err_strg[step, :] = out_pred_strg[step, :] .- CycleWorldUtils.oracle(env, parsed["horde"], parsed["gamma"])
    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    if !parsed["working"]
        save(savefile, results)
    else
        return results
    end
    # return results
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

