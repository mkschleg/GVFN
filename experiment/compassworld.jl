__precompile__(true)

module CompassWorldExperiment

using GVFN: CycleWorld, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
# using ProgressMeter
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
    end


    #Cycle world
    @add_arg_table as begin
        "--size"
        help="The size of the compass world chain"
        arg_type=Int64
        default=8
    end

    # Algorithms
    @add_arg_table as begin
        "--alg"
        help="Algorithm"
        default="TDLambda"
        "--luparams"
        help="Parameters"
        arg_type=Float64
        default=[0.5, 0.9]
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
        default=[0.5, 0.9]
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
build_features(state) = [[1.0]; state; 1.0.-state]




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

    pred_strg = zeros(num_steps, num_gvfs)
    err_strg = zeros(num_steps, 5)
    out_pred_strg = zeros(num_steps, 5)
    out_err_strg = zeros(num_steps, 5)

    out_horde = forward()
    out_opt = Descent(0.1)
    out_lu = TD()

    gvfn = GVFActionNetwork(num_gvfs, 3, 6*2 + 1, horde; init=(dims...)->0.01*randn(rng, Float32, dims...))
    model = Flux.Dense(num_gvfs, length(out_horde), Flux.σ)

    _, s_t = start!(env)
    state_list = [(1, zeros(6)) for t in 1:τ]
    popfirst!(state_list)
    push!(state_list, (1, build_features(s_t)))
    hidden_state_init = zeros(num_gvfs)

    for step in 1:num_steps
        print(step, "\r")

        a_t = get_action()
        # println(a_t)

        _, s_tp1, _, _ = step!(env, a_t[1])

        if length(state_list) == (τ+1)
            popfirst!(state_list)
        end
        push!(state_list, (a_t[1], build_features(s_tp1)))

        preds = train!(gvfn, opt, lu, hidden_state_init, state_list, s_tp1, a_t[1], a_t[2])
        train!(model, out_horde, out_opt, out_lu, preds, s_tp1, a_t[1], a_t[2])

        reset!(gvfn, hidden_state_init)
        preds = gvfn.(state_list)

        # if τ == 1
        #     reset!(gvfn, hidden_state_init)
        # else
        #     reset!(gvfn, preds[end-2])
        # end
        # reset!(gvfn, preds[end-2])
        out_preds = model.(preds)

        pred_strg[step, :] .= preds[end].data
        err_strg[step, :] .= preds[end].data[((8*collect(0:4)).+4)] .- oracle(env, "forward")
        out_pred_strg[step, :] .= out_preds[end].data
        out_err_strg[step, :] .= out_pred_strg[step, :] .- oracle(env, "forward")

        s_t .= s_tp1
        hidden_state_init .= Flux.data(preds[1])

        # println(env)
    end

    # results = Dict(["predictions"=>pred_strg, "error"=>err_strg])
    results = Dict(["predictions"=>pred_strg, "error"=>err_strg, "out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    save(savefile, results)
    # return pred_strg, err_strg
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

# CycleWorldExperiment.main_experiment()


