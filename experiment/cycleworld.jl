__precompile__(true)

module CycleWorldExperiment

using GVFN: CycleWorld, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter
using FileIO
using ArgParse
using Random


function arg_parse(as::ArgParseSettings = ArgParseSettings())

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
        "--rnn"
        help="whether we are using RNNs"
        action=:store_true
    end

    #Cycle world
    @add_arg_table as begin
        "--chain"
        help="The length of the cycle world chain"
        arg_type=Int64
        default=6
    end

    # Out Model Settings
    # @add_arg_table as begin
    #     "--"
    # end

    # shared settings
    @add_arg_table as begin
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
    end


    # RNN Settings
    @add_arg_table as begin
        "--rnnsize"
        help="Number of hidden units"
        arg_type=Int64
        default=6
        "--rnntype"
        help="Type of RNN"
        arg_type=String
        default="RNN"
    end

    # GVFN 
    @add_arg_table as begin
        "--gvfnalg"
        help="Algorithm"
        default="TDLambda"
        "--gvfnparams"
        help="Parameters"
        arg_type=Float64
        nargs='+'
        "--gvfnopt"
        help="Optimizer"
        default="Descent"
        "--gvfnoptparams"
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
    end

    return as
end


function onestep(chain_length::Integer)
    gvfs = [GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())]
    return Horde(gvfs)
end

function chain(chain_length::Integer)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())];
            [GVF(PredictionCumulant(i-1), ConstantDiscount(0.0), NullPolicy()) for i in 2:chain_length]]
    return Horde(gvfs)
end

function gamma_chain(chain_length::Integer, γ::AbstractFloat)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())];
            [GVF(PredictionCumulant(i-1), ConstantDiscount(0.0), NullPolicy()) for i in 2:chain_length];
            [GVF(FeatureCumulant(1), StateTerminationDiscount(0.9, ((env_state)->env_state[1] == 1)), NullPolicy())]]
    return Horde(gvfs)
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
    else
        throw("Bug Found")
    end

    return ret
end

build_features(s) = [1.0, s[1], 1-s[1]]

function main_experiment(args::Vector{String})

    as = arg_parse()
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
    horde = chain(parsed["chain"])
    if parsed["horde"] == "gamma_chain"
        horde = gamma_chain(parsed["chain"], parsed["gamma"])
    end

    num_gvfs = length(horde)

    alg_string = parsed["gvfnalg"]
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func(Float64.(parsed["gvfnparams"])...)
    τ=parsed["truncation"]

    opt_string = parsed["gvfnopt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["gvfnoptparams"])...)

    pred_strg = zeros(num_steps, num_gvfs)
    out_pred_strg = zeros(num_steps)
    err_strg = zeros(num_steps, num_gvfs)
    out_err_strg = zeros(num_steps)

    _, s_t = start!(env)

    gvfn = GVFNetwork(num_gvfs, 3, horde; init=(dims...)->0.001*randn(rng, Float32, dims...))
    model = Flux.Dense(num_gvfs, 1)
    out_horde = Horde([GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())])
    out_opt = Descent(0.1)
    out_lu = TD()

    state_list = [zeros(3) for t in 1:τ]
    popfirst!(state_list)
    push!(state_list, build_features(s_t))
    hidden_state_init = zeros(num_gvfs)

    @showprogress 0.1 "Step: " for step in 1:num_steps

        _, s_tp1, _, _ = step!(env, 1)

        if length(state_list) == (τ+1)
            popfirst!(state_list)
        end
        push!(state_list, build_features(s_tp1))

        preds = train!(gvfn, opt, lu, hidden_state_init, state_list, s_tp1)
        train!(model, out_horde, out_opt, out_lu, preds, s_tp1)

        out_preds = model.(preds)

        pred_strg[step, :] .= Flux.data(preds[end])
        err_strg[step, :] .= Flux.data(preds[end]) - oracle(env, parsed["horde"], parsed["gamma"])
        out_pred_strg[step] = out_preds[end].data[1]
        out_err_strg[step] = out_pred_strg[step][1] - oracle(env, parsed["horde"], parsed["gamma"])[1]

        s_t .= s_tp1
        hidden_state_init .= Flux.data(preds[1])
    end

    results = Dict(["predictions"=>pred_strg, "error"=>err_strg, "out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    save(savefile, results)
    # return pred_strg, err_strg
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

