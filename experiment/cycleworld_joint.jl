__precompile__(true)

module CycleWorldJointExperiment

using GVFN: CycleWorld, step!, start!
using GVFN
import Flux
import Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter
using FileIO
using Reproduce
# using ArgParse
using Random
using DataStructures: CircularBuffer

include("utils/util.jl")

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
        default=100
        "--verbose"
        action=:store_true
        "--working"
        action=:store_true
    end

    #Cycle world specific settings
    CycleWorldUtils.env_settings!(as)
    CycleWorldUtils.horde_settings!(as, "gvfn")
    CycleWorldUtils.horde_settings!(as, "out")

    # RNN
    FluxUtils.rnn_settings!(as)
    FluxUtils.opt_settings!(as)

    @add_arg_table as begin
        "--beta"
        help="learning update mixture"
        arg_type=Float64
        default=1.0 # Full RNN
    end

    return as
end

build_features(s) = Float32.([s[1], 1-s[1]])

function main_experiment(args::Vector{String})

    as = exp_settings()
    parsed = parse_args(args, as)
    save_loc=""
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        save_loc = get_save_loc(parsed["savefile"])
    end

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = CycleWorld(parsed["chain"])

    out_horde = CycleWorldUtils.get_horde(parsed, "out")
    gvfn_horde = CycleWorldUtils.get_horde(parsed, "gvfn")

    num_gvfs = length(out_horde)
    num_hidden = length(gvfn_horde)
    parsed["num_hidden"] = num_hidden

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)

    rnn = FluxUtils.construct_rnn(parsed["cell"], 2, length(gvfn_horde), Flux.σ)#; init=(dims...)->Float32(0.001)*randn(rng, Float32, dims...))
    out_model = Flux.Dense(num_hidden, length(out_horde), Flux.σ; initW=(dims...)->glorot_uniform(rng, dims...))

    h_state_strg = zeros(num_steps, num_hidden)
    h_state_err_strg = zeros(num_steps, num_hidden)
    out_pred_strg = zeros(num_steps, num_gvfs)
    out_err_strg = zeros(num_steps, num_gvfs)

    _, s_t = start!(env)

    state_list = CircularBuffer{Array{Float32, 1}}(τ+1)
    fill!(state_list, zeros(Float32, 2))
    push!(state_list, build_features(s_t))
    hidden_state_init = GVFN.get_initial_hidden_state(rnn)

    lu = GVFN.OnlineJointTD(parsed["beta"], state_list, hidden_state_init)

    for step in 1:num_steps
        if parsed["verbose"]
            if step % 1000 == 0
                print(step, "\r")
            end
        end

        _, s_tp1, _, _ = step!(env, 1)

        (preds, rnn_out) = train_step!(out_model, rnn, gvfn_horde, out_horde, opt, lu, build_features(s_tp1), s_tp1)

        h_state_strg[step,:] .= Flux.data(rnn_out[end])
        h_state_err_strg[step, :] .= h_state_strg[step, :] .- CycleWorldUtils.oracle(env, parsed["gvfnhorde"], parsed["gvfngamma"])

        out_pred_strg[step,:] .= Flux.data(preds[end])
        out_err_strg[step, :] .= out_pred_strg[step, :] .- CycleWorldUtils.oracle(env, parsed["outhorde"], parsed["outgamma"])
    end

    results = Dict("out_pred"=>out_pred_strg, "out_err"=>out_err_strg, "hidden_state_err"=>h_state_err_strg, "hidden_state"=>h_state_strg)

    if !parsed["working"]
        savefile=joinpath(save_loc, "results.jld2")
        save(savefile, results)
    else
        return results
    end
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

