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


function Flux.Optimise.apply!(o::Flux.RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(Flux.data(x))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + Flux.Optimise.ϵ)
end

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
        "--verbose"
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
        "--horde"
        help="The horde used for training"
        default="gamma_chain"
        "--gamma"
        help="The gamma value for the gamma_chain horde"
        arg_type=Float64
        default=0.9
    end

    # RNN
    @add_arg_table as begin
        "--opt"
        help="Optimizer"
        default="Descent"
        "--optparams"
        help="Parameters"
        arg_type=Float64
        default=[]
        nargs='+'
        "--cell"
        help="Cell"
        default="RNN"
        "--numhidden"
        help="Number of hidden units in cell"
        default=6
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
    elseif horde_str == "onestep"
        #TODO: Hack fix.
        tmp = zeros(chain_length + 1)
        tmp[chain_length - state] = 1
        ret = [tmp[1]]
    else
        throw("Bug Found")
    end

    return ret
end

build_features(s) = [1.0, s[1], 1-s[1]]

# function Flux.reset!(recur, hidden_state_init::Tuple)
#     Flux.reset!(recur)
#     for (idx, v) in enumerate(hidden_state_init)
#         recur.state[idx].data = v[idx]
#     end
# end

function main_experiment(args::Vector{String})

    # println("Start!")

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
    elseif parsed["horde"] == "onestep"
        horde = onestep(parsed["chain"])
    end

    τ=parsed["truncation"]
    num_gvfs = length(horde)

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["optparams"])...)

    cell_func = getproperty(Flux, Symbol(parsed["cell"]))
    rnn = cell_func(3, parsed["numhidden"])
    model = Flux.Chain(rnn, Flux.Dense(parsed["numhidden"], length(horde)))

    out_pred_strg = zeros(num_steps, num_gvfs)
    out_err_strg = zeros(num_steps, num_gvfs)

    _, s_t = start!(env)

    state_list = CircularBuffer{Array{Float64, 1}}(τ+1)
    fill!(state_list, zeros(3))
    push!(state_list, build_features(s_t))

    hidden_state_init = Flux.data.(rnn.cell(rnn.state, state_list[1]))

    for step in 1:num_steps

        # if parsed["verbose"]
        #     if step % 1000 == 0
        #         print(step, "\r")
        #     end
        # end
        _, s_tp1, _, _ = step!(env, 1)

        push!(state_list, build_features(s_tp1))
        # println(hidden_state_init)

        reset!(rnn, hidden_state_init[1])


        preds = model.(state_list)

        cumulants, discounts, π_prob = get(horde, s_tp1, Flux.data(preds[end]))

        δ = GVFN.tdloss(preds[end-1], cumulants, discounts, Flux.data(preds[end]))

        grads = Flux.Tracker.gradient(()->δ, Flux.params(model))

        for weights in Flux.params(model)
            Flux.Tracker.update!(opt, weights, -grads[weights])
        end

        reset!(rnn, hidden_state_init[1])

        preds = model.(state_list)

        out_pred_strg[step,:] = Flux.data(preds[end])
        out_err_strg[step, :] = out_pred_strg[step, :] .- oracle(env, parsed["horde"], parsed["gamma"])

        s_t .= s_tp1

        hidden_state_init = Flux.data.(rnn.cell(hidden_state_init[1], state_list[1]))

    end

    # println("Made it here!")
    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    save(savefile, results)
    # return pred_strg, err_strg
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

