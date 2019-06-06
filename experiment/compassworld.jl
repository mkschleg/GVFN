__precompile__(true)

module CompassWorldExperiment

using GVFN: CycleWorld, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter
# using FileIO
using JLD2
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
    end


    #Cycle world
    @add_arg_table as begin
        "--size"
        help="The size of the compass world chain"
        arg_type=Int64
        default=8
    end

    # GVFN
    @add_arg_table as begin
        "--alg"
        help="Algorithm"
        default="TDLambda"
        "--luparams"
        help="Parameters"
        arg_type=Float64
        default=[]
        nargs='+'
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
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
        "--activation"
        help="The activation used for the GVFN"
        arg_type=String
        default="sigmoid"
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
# build_features(state, action) = [[1.0]; state; 1.0.-state; onehot(3, action); 1.0.-onehot(3,action)]

function build_features(state, action)
    ϕ = [[1.0]; state; 1.0.-state]
    return [action==1 ? ϕ : zero(ϕ); action==2 ? ϕ : zero(ϕ); action==3 ? ϕ : zero(ϕ);]
end

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
    # println(args)
    # println(savefile)

    if savepath != ""
        if !isdir(savepath)
            mkpath(savepath)
        end
    end
    if isfile(savefile)
        return
    end

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = CompassWorld(parsed["size"], parsed["size"])
    num_state_features = get_num_features(env)
    horde = rafols()

    num_gvfs = length(horde)

    alg_string = parsed["alg"]
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func(Float64.(parsed["luparams"])...)
    τ=parsed["truncation"]

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["optparams"])...)

    # pred_strg = zeros(num_steps, num_gvfs)
    # err_strg = zeros(num_steps, 5)
    out_pred_strg = zeros(num_steps, 5)
    out_err_strg = zeros(num_steps, 5)

    act = Flux.σ
    if parsed["activation"] == "clamp"
        println("Clamp")
        # act = (x)->clamp(x, 0.0, 1.0)
        act = clip
    end

    out_horde = forward()
    out_opt = Descent(0.1)
    out_lu = TD()
    model = SingleLayer(num_gvfs, length(out_horde), sigmoid, sigmoid′)

    _, s_t = start!(env)
    ϕ = build_features(s_t, 1)
    state_list = CircularBuffer{typeof(ϕ)}(τ+1)

    gvfn = GVFNetwork(num_gvfs, length(ϕ), horde; init=(dims...)->0.0001.*(rand(rng, Float64, dims...).-0.5), σ_int=act)

    hidden_state_init = zeros(num_gvfs)

    action_state = ""
    action_state, a_tm1 = get_action(action_state, s_t, rng)
    fill!(state_list, zero(ϕ))
    push!(state_list, build_features(s_t, a_tm1[1]))

    # @showprogress 0.1 "Step: " for step in 1:num_steps
    for step in 1:num_steps
        # print(step, "\r")
        action_state, a_t = get_action(action_state, s_t, rng)
        # a_t = get_action()

        _, s_tp1, _, _ = step!(env, a_t[1])

        push!(state_list, build_features(s_tp1, a_t[1]))

        preds = train!(gvfn, opt, lu, hidden_state_init, state_list, s_tp1, a_t[1], a_t[2])
        # preds =
        reset!(gvfn, hidden_state_init)
        preds = gvfn.(state_list)
        train!(model, out_horde, out_opt, out_lu, Flux.data.(preds), s_tp1, a_t[1], a_t[2])

        out_preds = model(preds[end])

        out_pred_strg[step, :] .= Flux.data(out_preds)
        out_err_strg[step, :] .= out_pred_strg[step, :] .- oracle(env, "forward")

        s_t .= s_tp1
        hidden_state_init .= Flux.data(preds[1])

    end

    results = Dict(["out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    # save(savefile, results)
    JLD2.save(
        JLD2.FileIO.File(JLD2.FileIO.DataFormat{:JLD2},
                         savefile),
        Dict("results"=>results); compress=true)
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

# CycleWorldExperiment.main_experiment()


