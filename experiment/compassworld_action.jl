__precompile__(true)

module CompassWorldActionExperiment

using GVFN: CompassWorld, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter
# using FileIO
using JLD2
using Reproduce
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
        "--working"
        action=:store_true
        "--verbose"
        action=:store_true
    end


    #Compass World
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
        "--params"
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
        "--act"
        help="The activation used for the GVFN"
        arg_type=String
        default="sigmoid"
        "--feature"
        help="The feature creator to use"
        arg_type=String
        default="standard"
    end

    return as
end


# function oracle(env::CompassWorld, horde_str)
#     cwc = GVFN.CompassWorldConst
#     state = env.agent_state
#     ret = Array{Float64,1}()
#     if horde_str == "forward"
#         ret = zeros(5)
#         if state.dir == cwc.NORTH
#             ret[cwc.ORANGE] = 1
#         elseif state.dir == cwc.SOUTH
#             ret[cwc.RED] = 1
#         elseif state.dir == cwc.WEST
#             if state.y == 1
#                 ret[cwc.GREEN] = 1
#             else
#                 ret[cwc.BLUE] = 1
#             end
#         elseif state.dir == cwc.EAST
#             ret[cwc.YELLOW] = 1
#         else
#             println(state.dir)
#             throw("Bug Found in Oracle:Forward")
#         end
#     elseif horde_str == "rafols"
#         throw("Not Implemented...")
#     else
#         throw("Bug Found in Oracle")
#     end

#     return ret
# end


function main_experiment(args::Vector{String})

    cwu = GVFN.CompassWorldUtils

    as = arg_parse()
    parsed = parse_args(args, as)

    ######
    # Experiment Setup
    ######
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

    out_pred_strg = zeros(num_steps, 5)
    out_err_strg = zeros(num_steps, 5)

    ######
    # Environment Setup
    ######
    
    env = CompassWorld(parsed["size"], parsed["size"])
    num_state_features = get_num_features(env)

    _, s_t = start!(env) # Start environment

    #####
    # Agent specific setup.
    #####
    
    horde = cwu.get_horde(parsed)
    out_horde = cwu.forward()

    fc = cwu.StandardFeatureCreator()
    if parsed["feature"] == "action"
        fc = cwu.ActionTileFeatureCreator()
    end

    fs = JuliaRL.FeatureCreators.feature_size(fc)

    ap = cwu.ActingPolicy()
    
    agent = GVFN.GVFNActionAgent(horde, out_horde,
                                 fc, fs,
                                 3,
                                 ap,
                                 parsed;
                                 rng=rng,
                                 init_func=(dims...)->glorot_uniform(rng, dims...))
    
    action = start!(agent, s_t; rng=rng) # Start agent
    verbose = parsed["verbose"]
    
    @showprogress 0.1 "Step: " for step in 1:num_steps

        _, s_tp1, _, _ = step!(env, action)
        out_preds, action = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step, :] .= Flux.data(out_preds)
        out_err_strg[step, :] .= out_pred_strg[step, :] .- cwu.oracle(env, "forward")

        if verbose
            println("step: $(step)")
            println(env)
            println(agent)
            println(out_preds)
        end

        
    end

    results = Dict(["rmse"=>sqrt.(mean(out_err_strg.^2; dims=2))])
    if !parsed["working"]
        JLD2.@save savefile results
    else
        return results, out_pred_strg, out_err_strg
    end
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

# CycleWorldExperiment.main_experiment()


