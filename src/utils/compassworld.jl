module CompassWorldUtils

using ..GVFN, Reproduce

using JuliaRL.FeatureCreators
using Random

cwc = GVFN.CompassWorldConst

function env_settings!(as::ArgParseSettings)
    @add_arg_table as begin
        "--size"
        help="The size of the compass world"
        arg_type=Int64
        default=8
    end
end

function horde_settings!(as::ArgParseSettings, prefix::AbstractString="")
    add_arg_table(as,
                  "--$(prefix)horde",
                  Dict(:help=>"The horde used for training",
                       :default=>"rafols"))
end

function rafols(pred_offset::Integer=0)

    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.FORWARD)),
                    GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    GVF(FeatureCumulant(color), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(8*(color-1) + 4 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    GVF(PredictionCumulant(8*(color-1) + 4 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    GVF(PredictionCumulant(8*(color-1) + 5 + pred_offset), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(8*(color-1) + 6 + pred_offset), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD))]
        append!(gvfs, new_gvfs)
    end
    return Horde(gvfs)
end

function forward()
    cwc = GVFN.CompassWorldConst
    gvfs = [GVF(FeatureCumulant(color),
                StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)),
                PersistentPolicy(cwc.FORWARD)) for color in 1:5]
    return Horde(gvfs)
end

function onestep()
    cwc = GVFN.CompassWorldConst
    gvfs = [GVF(FeatureCumulant(color),
                ConstantDiscount(0.0),
                PersistentPolicy(cwc.FORWARD)) for color in 1:5]
    return Horde(gvfs)
end

function gammas(gammas = [collect(0.0:0.05:0.95); [0.975, 0.99]])
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(
            FeatureCumulant(color),
            ConstantDiscount(γ),
            PersistentPolicy(cwc.FORWARD)) for γ in gammas]
        append!(gvfs, new_gvfs)
        # new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(γ, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for γ in 0.0:0.05:0.95]
    end
    return Horde(gvfs)
end

function gammas_term(gammas = [collect(0.0:0.05:0.95); [0.975, 0.99]])
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(
            FeatureCumulant(color),
            StateTerminationDiscount(γ, ((env_state)->env_state[cwc.WHITE] == 0)),
            PersistentPolicy(cwc.FORWARD)) for γ in gammas]
        append!(gvfs, new_gvfs)
    end
    return Horde(gvfs)
end

function gammas_scaled(gammas = [collect(0.0:0.05:0.95); [0.975, 0.99]])
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(
            ScaledCumulant(1-γ, FeatureCumulant(color)),
            ConstantDiscount(γ),
            PersistentPolicy(cwc.FORWARD)) for γ in gammas]
        append!(gvfs, new_gvfs)
        # new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(γ, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for γ in 0.0:0.05:0.95]
    end
    return Horde(gvfs)
end

function gammas_with_scaled_white(gammas = [collect(0.0:0.05:0.95); [0.975, 0.99]])
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(
            FeatureCumulant(color),
            ConstantDiscount(γ),
            PersistentPolicy(cwc.FORWARD)) for γ in gammas]
        append!(gvfs, new_gvfs)
        # new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(γ, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for γ in 0.0:0.05:0.95]
    end
    new_gvfs = [GVF(
        ScaledCumulant(1-γ, FeatureCumulant(cwc.WHITE)),
        ConstantDiscount(γ),
        PersistentPolicy(cwc.FORWARD)) for γ in gammas]
    return Horde(gvfs)
end

function gammas_and_expert(gammas = [collect(0.0:0.05:0.95); [0.975, 0.99]], pred_offset=0)
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(8*(color-1) + 1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    GVF(PredictionCumulant(8*(color-1) + 1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    GVF(PredictionCumulant(8*(color-1) + 2 + pred_offset), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(8*(color-1) + 3 + pred_offset), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD))]
        append!(gvfs, new_gvfs)
    end
    for color in 1:5
        new_gvfs = [GVF(
            FeatureCumulant(color),
            ConstantDiscount(γ),
            PersistentPolicy(cwc.FORWARD)) for γ in gammas]
        append!(gvfs, new_gvfs)
        # new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(γ, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for γ in 0.0:0.05:0.95]
    end
    new_gvfs = [GVF(
        ScaledCumulant(1-γ, FeatureCumulant(cwc.WHITE)),
        ConstantDiscount(γ),
        PersistentPolicy(cwc.FORWARD)) for γ in gammas]
    append!(gvfs, new_gvfs)
    return Horde(gvfs)
end

function direction_conditional(gammas = [collect(0.0:0.05:0.95); [0.975, 0.99]], pred_offset=0)
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD))]
        append!(gvfs, new_gvfs)
    end
    for color in 1:5
        new_gvfs = [GVF(
            FeatureCumulant(color),
            ConstantDiscount(γ),
            PersistentPolicy(cwc.FORWARD)) for γ in gammas]
        append!(gvfs, new_gvfs)
    end

    for color in 1:5
        new_gvfs = [GVF(
            FeatureCumulant(color),
            ConstantDiscount(γ),
            PredictionConditionalPolicy(
                PersistentPolicy(cwc.FORWARD),
                (preds_tp1)->(preds_tp1[color + pred_offset]>=0.7))) for γ in [0.0, 0.5, 0.9, 0.95]]
        append!(gvfs, new_gvfs)
    end
    # new_gvfs = [GVF(
    #     ScaledCumulant(1-γ, FeatureCumulant(cwc.WHITE)),
    #     ConstantDiscount(γ),
    #     PersistentPolicy(cwc.FORWARD)) for γ in gammas]
    return Horde(gvfs)
end

function test_network(pred_offset::Integer=0)
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.FORWARD)),
                    GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    GVF(FeatureCumulant(color), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    # GVF(PredictionCumulant(7*(color-1) + 1), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(6*(color-1) + 2 + pred_offset), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(6*(color-1) + 3 + pred_offset), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD))]
                    # GVF(PredictionCumulant(8*(color-1) + 4), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    # GVF(PredictionCumulant(8*(color-1) + 4), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    # GVF(PredictionCumulant(8*(color-1) + 5), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    # GVF(PredictionCumulant(8*(color-1) + 6), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD))]
        append!(gvfs, new_gvfs)
    end
    println(length(gvfs))
    return Horde(gvfs)
end


function get_horde(horde_str::AbstractString, pred_offset::Integer=0)
    horde = forward()
    if horde_str == "forward"
        horde = forward()
    elseif horde_str == "onestep"
        horde = onestep()
    elseif horde_str == "rafols"
        horde = rafols(pred_offset)
    elseif horde_str == "out_gammas"
        horde = gammas_scaled(0.7)
    elseif horde_str == "gammas"
        horde = gammas_term()
    elseif horde_str == "gammas_scaled"
        horde = gammas_scaled()
    elseif horde_str == "aj_gammas"
        horde = gammas(1.0 .- 2.0 .^ collect(-7:-1))
    elseif horde_str == "aj_gammas_scaled"
        horde = gammas_scaled(1.0 .- 2.0 .^ collect(-7:-1))
    elseif horde_str == "aj_gammas_term"
        horde = gammas_term(1.0 .- 2.0 .^ collect(-7:-1))
    elseif horde_str == "gammas_with_scaled_white"
        horde = gammas_with_scaled_white(1.0 .- 2.0 .^ collect(-7:-1))
    elseif horde_str == "gammas_and_expert"
        horde = gammas_and_expert(1.0 .- 2.0 .^ collect(-7:-1))
    elseif horde_str == "direction_conditional"
        horde = direction_conditional(1.0 .- 2.0 .^ collect(-7:-1))
    elseif horde_str == "test"
        horde = test_network(pred_offset)
    else
        throw("Unknown horde")
    end
    return horde
end

get_horde(parsed::Dict, prefix::AbstractString="", pred_offset::Integer=0) = get_horde(parsed["$(prefix)horde"], pred_offset)

function oracle_onestep(state, world_dims)
    ret = zeros(5)

    # Forward
    if state.dir == cwc.NORTH
        # Orange Check
        ret[cwc.ORANGE] = (state.y == 1 || state.y == 2)
    elseif state.dir == cwc.SOUTH
        # Red Check
        ret[cwc.RED] = (state.y == world_dims.height || state.y == world_dims.height - 1)
    elseif state.dir == cwc.WEST
        if state.y == 1
            # Green Check
            ret[cwc.GREEN] = (state.x == 1 || state.x == 2)
        else
            # Blue Check
            ret[cwc.BLUE] = (state.x == 1 || state.x == 2)
        end
    elseif state.dir == cwc.EAST
        # Yellow Check
        ret[cwc.YELLOW] = (state.x == world_dims.width || state.x == world_dims.width - 1)
    else
        throw("OH NO! oracle_onestep $(state.dir)")
    end
    # 
    ret
end

function oracle_forward(state)
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
    return ret
end

function oracle_rafols(state)
    throw("Rafols not implemented.")
end

function oracle_gammas(state)
    throw("Gammas not implemented.")
end

function oracle_out_gammas(state, world_dims)

    γ = 0.7
    
    ret = zeros(5)
    if state.dir == cwc.NORTH
        # Orange
        ret[cwc.ORANGE] = γ^(max(0, state.y-2))
    elseif state.dir == cwc.SOUTH
        # Red
        ret[cwc.RED] = γ^(max(0, world_dims.height - state.y - 1))
    elseif state.dir == cwc.WEST
        if state.y == 1
            # Green
            ret[cwc.GREEN] = γ^(max(0, state.x-2))
        else
            #Blue
            ret[cwc.BLUE] = γ^(max(0, state.x-2))
        end
    elseif state.dir == cwc.EAST
        #Yellow
        ret[cwc.YELLOW] = γ^(max(0, world_dims.width - state.x - 1))
    else
        println(state.dir)
        throw("Bug Found in Oracle:Forward")
    end
    return ret
    
end

function oracle(env::CompassWorld, horde_str)
    
    state = env.agent_state
    ret = Array{Float64,1}()
    if horde_str == "forward"
        ret = oracle_forward(state)
    elseif horde_str == "onestep"
        ret = oracle_onestep(state, env.world_dims)
    elseif horde_str == "rafols"
        ret = oracle_rafols(state)
    elseif horde_str == "gammas"
        ret = oracle_gammas(state)
    elseif horde_str == "out_gammas"
        ret = oracle_out_gammas(state, env.world_dims)
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

mutable struct ActingPolicy <: GVFN.AbstractActingPolicy
    state::String
    ActingPolicy() = new("")
end

function (π::ActingPolicy)(state_t, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    s, action = get_action(π.state, state_t, rng)
    π.state = s
    return action[1], action[1]
end


function get_behavior_policy(policy_str)

    if policy_str == "rafols"
        ap = ActingPolicy()
    elseif policy_str == "forward"
        ap = GVFN.RandomActingPolicy([1/4, 1/4, 1/2])
    elseif policy_str == "random"
        ap = GVFN.RandomActingPolicy([1/3, 1/3, 1/3])
    else
        throw("Unknown behavior policy")
    end
end




onehot(size, idx) = begin; a=zeros(size);a[idx] = 1.0; return a end;
build_features(state, action) = [[1.0]; state; 1.0.-state; onehot(3, action); 1.0.-onehot(3,action)]
# build_features_action(state, action) = [[1.0]; state; 1.0.-state; onehot(3, action); 1.0.-onehot(3,action)]

function build_features_action(state, action)
    ϕ = [[1.0]; state; 1.0.-state]
    return [action==1 ? ϕ : zero(ϕ); action==2 ? ϕ : zero(ϕ); action==3 ? ϕ : zero(ϕ);]
end

mutable struct StandardFeatureCreator end

(fc::StandardFeatureCreator)(s, a) = create_features(fc, s, a)
JuliaRL.FeatureCreators.create_features(fc::StandardFeatureCreator, state, action) = [[1.0]; state; 1.0.-state; onehot(3, action); 1.0.-onehot(3,action)]
JuliaRL.FeatureCreators.feature_size(fc::StandardFeatureCreator) = 19

mutable struct ActionTileFeatureCreator end

(fc::ActionTileFeatureCreator)(s, a) = create_features(fc, s, a)
function JuliaRL.FeatureCreators.create_features(fc::ActionTileFeatureCreator, state, action)
    ϕ = [[1.0]; state; 1.0.-state]
    return [action==1 ? ϕ : zero(ϕ); action==2 ? ϕ : zero(ϕ); action==3 ? ϕ : zero(ϕ);]
end
JuliaRL.FeatureCreators.feature_size(fc::ActionTileFeatureCreator) = 39

end
