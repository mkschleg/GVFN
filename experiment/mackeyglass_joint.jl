__precompile__(true)

module MackeyGlassJointExperiment

using GVFN: MackeyGlass, step!, start!
using GVFN
import Flux
import Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter
using FileIO, JLD2
using Reproduce
using Random
using DataStructures: CircularBuffer

import GVFN.FluxUtils

# include("utils/util.jl")

function Flux.Optimise.apply!(o::Flux.RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(Flux.data(x))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + Flux.Optimise.ϵ)
end

function exp_settings(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))

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
        default=100000
        "--horizon"
        help="prediction horizon"
        arg_type=Int64
        default=12
        "--verbose"
        action=:store_true
        "--working"
        action=:store_true
    end

    # MG Specific settings
    TimeSeriesUtils.horde_settings!(as)

    # RNN
    FluxUtils.rnn_settings!(as)
    FluxUtils.opt_settings!(as)

    @add_arg_table as begin
        "--beta"
        help="learning update mixture"
        arg_type=Float64
        default=1.0 # Full RNN
        "--act"
        help="Activation function for rnn layer"
        arg_type=String
        default="sigmoid"
    end

    return as
end

function main_experiment(args::Vector{String})

    as = exp_settings()
    parsed = parse_args(args, as)
    save_loc=""
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        save_loc = Reproduce.get_save_dir(parsed)
        if isfile(joinpath(save_loc, "results.jld2"))
            return
        end
    end


    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = MackeyGlass()

    # TODO
    # apparently need an output horde and a gvfn horde?
    TimeSeriesUtils.get_horde(parsed)

    num_gvfs = length(out_horde)
    num_hidden = length(gvfn_horde)
    parsed["num_hidden"] = num_hidden

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)

    rnn = FluxUtils.construct_rnn("RNN", 1, 1)
    if parsed["cell"] == "RNN"
        act = FluxUtils.get_activation(parsed["act"])
        rnn = FluxUtils.construct_rnn(parsed["cell"], 2, length(gvfn_horde), act)#; init=(dims...)->Float32(0.001)*randn(rng, Float32, dims...))
    else
        rnn = FluxUtils.construct_rnn(parsed["cell"], 2, length(gvfn_horde))#; init=(dims...)->Float32(0.001)*randn(rng, Float32, dims...))
    end
    out_model = Flux.Dense(num_hidden, length(out_horde), Flux.σ; initW=(dims...)->glorot_uniform(rng, dims...))

    out_pred_strg = zeros(num_steps, num_gvfs)

    _, s_t = start!(env)

    state_list = CircularBuffer{Array{Float32, 1}}(τ+1)
    fill!(state_list, zeros(Float32, 2))
    push!(state_list, s_t)
    hidden_state_init = GVFN.get_initial_hidden_state(rnn)

    lu = GVFN.OnlineJointTD(parsed["beta"], state_list, hidden_state_init)

    for step in 1:num_steps
        if parsed["verbose"]
            if step % 1000 == 0
                print(step, "\r")
            end
        end

        _, s_tp1, _ = step!(env, 1)

        (preds, rnn_out) = train_step!(out_model, rnn, gvfn_horde, out_horde, opt, lu, s_tp1, s_tp1)

        out_pred_strg[step,:] .= Flux.data(preds[end])
    end

    results = Dict("predictions"=>out_pred_strg)

    if !parsed["working"]
        savefile=joinpath(save_loc, "results.jld2")
        # save(savefile, results)
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

