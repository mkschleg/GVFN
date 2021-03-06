__precompile__(true)

module CompassWorldRNNATExperiment

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
using Reproduce
# using Reproduce
using Random

using Flux.Tracker: TrackedArray, TrackedReal, track, @grad


using DataStructures: CircularBuffer

# function Flux.Optimise.apply!(o::Flux.RMSProp, x, Δ)
#   η, ρ = o.eta, o.rho
#   acc = get!(o.acc, x, zero(x))::typeof(Flux.data(x))
#   @. acc = ρ * acc + (1 - ρ) * Δ^2
#   @. Δ *= η / (√acc + Flux.Optimise.ϵ)
# end

const cwu = GVFN.CompassWorldUtils
const FLU = GVFN.FluxUtils

function arg_parse(as::ArgParseSettings = ArgParseSettings())

    #Experiment

    GVFN.exp_settings!(as)


    #Compass world settings
    @add_arg_table as begin
        "--size"
        help="The size of the compass world chain"
        arg_type=Int64
        default=8
    end

    # # shared settings
    # @add_arg_table as begin
    #     "--truncation", "-t"
    #     help="Truncation parameter for bptt"
    #     arg_type=Int64
    #     default=1
    # end

    cwu.horde_settings!(as, "aux")
    FLU.opt_settings!(as)

    # shared settings
    FLU.rnn_settings!(as)    

    # RNN Settings
    @add_arg_table as begin
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



    #####
    # Setup experiment environment
    #####
    as = arg_parse()
    parsed = parse_args(args, as)
    parsed["prev_action_or_not"] = true

    savepath = ""
    savefile = ""
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, "results.jld2")
        if isfile(savefile)
            println("Here")
            return
        end
    end

    ####
    # General Experiment parameters
    ####
    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = CompassWorld(parsed["size"], parsed["size"])
    num_state_features = get_num_features(env)


    _, s_t = start!(env)

    out_horde = cwu.forward()
    aux_horde = cwu.get_horde(parsed, "aux", length(out_horde))

    out_pred_strg = zeros(num_steps, length(out_horde))
    out_err_strg = zeros(num_steps, length(out_horde))

    fc = cwu.StandardFeatureCreator()
    if parsed["feature"] == "action"
        fc = cwu.ActionTileFeatureCreator()
    end

    fs = JuliaRL.FeatureCreators.feature_size(fc)

    ap = cwu.ActingPolicy()
    
    # agent = RNNAgent(parsed; rng=rng)
    agent = GVFN.RNNAgent(Horde([out_horde.gvfs; aux_horde.gvfs]),
                          fc, fs, ap, parsed;
                          rng=rng,
                          init_func=(dims...)->glorot_uniform(rng, dims...))
    action = start!(agent, s_t; rng=rng)

    # @showprogress 0.1 "Step: " for step in 1:num_steps
    for step in 1:num_steps
        if step%100000 == 0
            # println("Garbage Clean!")
            GC.gc()
        end
        if parsed["verbose"]
            if step%10000 == 0
                print(step, "\r")
            end
        end

        _, s_tp1, _, _ = step!(env, action)
        out_preds, action = step!(agent, s_tp1, 0, false; rng=rng)

        out_pred_strg[step, :] .= Flux.data.(out_preds[1:length(out_horde)])
        out_err_strg[step, :] .= out_pred_strg[step, :] .- cwu.oracle(env, "forward")
        # println(out_pred_strg[step, :])
    end

    results = Dict(["rmse"=>sqrt.(mean(out_err_strg.^2; dims=2))])
    if !parsed["working"]
        JLD2.@save savefile results
    else
        return out_pred_strg, out_err_strg, results
    end
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end
