module FourRoomsExperiment

using GVFN: ContFourRooms, step!, start!
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

using JuliaRL.FeatureCreators

function arg_parse(as::ArgParseSettings = ArgParseSettings(exc_handler=ArgParse.debug_handler))

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


function gammas_scaled()
    cfrp = GVFN.ContFourRoomsParams
    gvfs = Array{GVF, 1}()
    for side in 3:6
        new_gvfs = vcat([[GVF(
            ScaledCumulant(2.0^-k, FeatureCumulant(side)),
            ConstantDiscount(1-2.0^-k),
            PersistentPolicy(action)) for k in 1:7] for action in cfrp.ACTIONS]...)
        append!(gvfs, new_gvfs)
        # new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(γ, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for γ in 0.0:0.05:0.95]
    end
    return Horde(gvfs)
end

function one_step()
    cfrp = GVFN.ContFourRoomsParams
    # gvfs = Array{GVF, 1}()
    gvfs = [GVF(
        FeatureCumulant(side),
        ConstantDiscount(0.9),
        GVFN.RandomPolicy(fill(0.25, 4))) for side in 3:6]
    # for side in 1:4

    #     # new_gvfs = vcat([[GVF(
    #     #     ScaledCumulant(2^-7, FeatureCumulant(side)),
    #     #     ConstantDiscount(1-2^-7),
    #     #     PersistentPolicy(cfrp.RIGHT)) for k in [0]] for action in cfrp.ACTIONS]...)
    #     append!(gvfs, new_gvfs)
    #     # new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(γ, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for γ in 0.0:0.05:0.95]
    # end
    return Horde(gvfs)
end


function main_experiment(args::Vector{String})



    as = arg_parse()
    parsed = parse_args(args, as)

    ######
    # Experiment Setup
    ######
    # savepath = ""
    # savefile = ""
    # if !parsed["working"]
    #     create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
    #     savepath = Reproduce.get_save_dir(parsed)
    #     savefile = joinpath(savepath, "results.jld2")
    #     if isfile(savefile)
    #         return
    #     end
    # end

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    out_pred_strg = zeros(num_steps, 4)

    

    ######
    # Environment Setup
    ######
    
    env = ContFourRooms(;normalize=true)
    num_state_features = 2

    _, s_t = start!(env) # Start environment

    state_strg = Array{typeof(s_t), 1}(undef, num_steps+1)
    
    #####
    # Agent specific setup.
    #####
    
    horde = gammas_scaled()
    out_horde = one_step()

    onehot(size, idx) = begin; a=zeros(size);a[idx] = 1.0; return a end;
    
    num_tilings = 64
    num_tiles = 8
    tilecoder = TileCoder(num_tilings, num_tiles, 2)
    num_features = num_tilings*(num_tiles+1)^2
    println(num_features)
    fc=(state, action)->begin; x=zeros(num_features); x[create_features(tilecoder, state[1:2])] .= 1; x; end;
    
    # fc = cwu.StandardFeatureCreator()
    # if parsed["feature"] == "action"
    #     fc = cwu.ActionTileFeatureCreator()
    # end

    # fs = JuliaRL.FeatureCreators.feature_size(fc)
    # fs = 13

    ap = GVFN.RandomActingPolicy(fill(0.25, 4))
    
    agent = GVFN.GVFNAgent(horde, out_horde,
                           fc, num_features,
                           ap,
                           parsed;
                           rng=rng,
                           init_func=(dims...)->0.001f0*glorot_uniform(rng, dims...))
    
    action = start!(agent, s_t[1]; rng=rng) # Start agent

    
    @showprogress 0.1 "Step: " for step in 1:num_steps

        _, s_tp1, _, _ = step!(env, action)
        
        out_preds, action = step!(agent, s_tp1[1], 0, false; rng=rng)

        out_pred_strg[step, :] .= Flux.data(out_preds)
        state_strg[step+1] = copy.(s_tp1)

    end

    return out_pred_strg, state_strg
    # if !parsed["working"]
    #     JLD2.@save savefile results
    # else
    #     return results
    # end
end


end



