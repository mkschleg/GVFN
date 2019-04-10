__precompile__(true)

module CompassWorldRNNExperiment

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

using Flux.Tracker: TrackedArray, TrackedReal, track, @grad


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


    #Compass world settings
    @add_arg_table as begin
        "--size"
        help="The size of the compass world chain"
        arg_type=Int64
        default=8
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
    end

    # RNN Settings
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
        default="RNNCell"
        "--numhidden"
        help="Number of hidden units in cell"
        default=45
    end

    return as
end


function rafols()
    
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.FORWARD)),
                    GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    GVF(FeatureCumulant(color), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(8*(color-1) + 4), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    GVF(PredictionCumulant(8*(color-1) + 4), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    GVF(PredictionCumulant(8*(color-1) + 5), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(8*(color-1) + 6), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD))]
        append!(gvfs, new_gvfs)
    end
    return Horde(gvfs)
end

function forward()
    cwc = GVFN.CompassWorldConst
    gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for color in 1:5]
    return Horde(gvfs)
end

function oracle(env::CompassWorld, horde_str)
    cwc = GVFN.CompassWorldConst
    state = env.agent_state
    ret = Array{Float64,1}()
    if horde_str == "forward"
        ret = zeros(5)
        if state.dir == cwc.NORTH
            ret[cwc.ORANGE] = 1
        elseif state.dir == cwc.SOUTH
            ret[cwc.RED] = 1
        elseif state.dir == cwc.WEST
            if state.y == 1
                ret[cwc.GREEN] = 1
            else
                ret[cwc.BLUE] = 1
            end
        elseif state.dir == cwc.EAST
            ret[cwc.YELLOW] = 1
        else
            println(state.dir)
            throw("Bug Found in Oracle:Forward")
        end
    elseif horde_str == "rafols"
        throw("Not Implemented...")
    else
        throw("Bug Found in Oracle")
    end

    return ret
end


function get_action(state, env_state, rng=Random.GLOBAL_RNG)

    if state == ""
        state = "Random"
    end

    cwc = GVFN.CompassWorldConst
    

    if state == "Random"
        r = rand(rng)
        if r > 0.9
            state = "Leap"
        end
    end

    if state == "Leap"
        if env_state[cwc.WHITE] == 0.0
            state = "Random"
        else
            return state, (cwc.FORWARD, 1.0)
        end
    end
    r = rand(rng)
    if r < 0.2
        return state, (cwc.RIGHT, 0.2)
    elseif r < 0.4
        return state, (cwc.LEFT, 0.2)
    else
        return state, (cwc.FORWARD, 0.6)
    end
end

function get_action(rng=Random.GLOBAL_RNG)
    
    cwc = GVFN.CompassWorldConst
    r = rand(rng)
    if r < 0.2
        return cwc.RIGHT, 0.2
    elseif r < 0.4
        return cwc.LEFT, 0.2
    else
        return cwc.FORWARD, 0.6
    end
end

# build_features(state) = state
onehot(size, idx) = begin; a=zeros(size);a[idx] = 1.0; return a end;
build_features(state, action) = [[1.0]; state; 1.0.-state; onehot(3, action); 1.0.-onehot(3,action)]

# Flux.σ(x::AbstractArray) = Flux.σ.(x)

function clip(a)
    clamp.(a, 0.0, 1.0)
end

function clip(a::TrackedArray)
    track(clip, a)
end
@grad function clip(a)
    return clip(Flux.data(a)), Δ -> Tuple(Δ)
end


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

    env = CompassWorld(parsed["size"], parsed["size"])
    num_state_features = get_num_features(env)

    horde = rafols()
    if parsed["horde"] == "forward"
        horde = forward()
    end

    num_gvfs = length(horde)

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["optparams"])...)
    τ=parsed["truncation"]

    # pred_strg = zeros(num_steps, num_gvfs)
    # err_strg = zeros(num_steps, 5)
    out_pred_strg = zeros(num_steps, num_gvfs)
    out_err_strg = zeros(num_steps, num_gvfs)

    _, s_t = start!(env)
    ϕ = build_features(s_t, 1)


    cell_func = getproperty(Flux, Symbol(parsed["cell"]))
    rnn = cell_func(length(ϕ), parsed["numhidden"])
    model = Flux.Chain(rnn, Flux.Dense(parsed["numhidden"], length(horde)))
    # rnn = Flux.RNN(length(ϕ), parsed["numhidden"])
    # model = Flux.Chain(rnn, Flux.Dense(parsed["numhidden"], length(horde)))

    hidden_state_init = Flux.data(Flux.hidden(rnn.cell))
    # println(hidden_state_init)
    state_list = CircularBuffer{typeof(ϕ)}(τ+1)

    action_state = ""
    action_state, a_tm1 = get_action(action_state, s_t, rng)
    fill!(state_list, zero(ϕ))
    push!(state_list, build_features(s_t, a_tm1[1]))

    # hidden_state_init = Flux.data.(rnn.cell(hidden_state_init, state_list[1]))
    # println(size(rnn.state), " ", size(state_list[1]))
    hidden_state_init = nothing

    if parsed["cell"] == "LSTM"
        hidden_state_init = Flux.data.(rnn.cell(rnn.state, state_list[1])[1])
    else
        hidden_state_init = Flux.data(rnn.cell(rnn.state, state_list[1])[1])
    end

    ρ = zeros(Float32, num_gvfs)
    cumulants = zeros(Float32, num_gvfs)
    discounts = zeros(Float32, num_gvfs)
    π_prob = zeros(Float32, num_gvfs)
    preds_tilde = zeros(Float32, num_gvfs)

    preds = model.(state_list)


    for step in 1:num_steps
        if step%100000 == 0
            println("Garbage Clean!")
            GC.gc()
        end
        if parsed["verbose"]
            if step%10000 == 0
                print(step, "\r")
            end
        end
        action_state, a_t = get_action(action_state, s_t, rng)

        _, s_tp1, _, _ = step!(env, a_t[1])

        push!(state_list, build_features(s_tp1, a_t[1]))


        reset!(rnn, hidden_state_init)


        preds .= model.(state_list)
        preds_tilde .= Flux.data(preds[end])

        get!(cumulants, discounts, π_prob, horde, a_t[1], s_tp1, preds_tilde)

        ρ .= π_prob./a_t[2]

        grads = Flux.Tracker.gradient(()->GVFN.offpolicy_tdloss(ρ, preds[end-1], cumulants, discounts, preds_tilde), Flux.params(model))

        for weights in Flux.params(model)
            Flux.Tracker.update!(opt, weights, -grads[weights])
        end

        reset!(rnn, hidden_state_init)
        Flux.truncate!(rnn)
        preds .= model.(state_list)

        out_pred_strg[step, :] .= Flux.data(preds[end])
        out_err_strg[step, :] .= out_pred_strg[step, :] .- oracle(env, "forward")

        s_t .= s_tp1

        if parsed["cell"] == "LSTM"
            hidden_state_init = Flux.data.(rnn.cell(hidden_state_init, state_list[1])[1])
        else
            hidden_state_init = Flux.data(rnn.cell(hidden_state_init, state_list[1])[1])
        end
        # println(hidden_state_init)

    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    save(savefile, results)
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

# CycleWorldExperiment.main_experiment()


