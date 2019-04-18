module CompassWorldUtils

using GVFN, Reproduce

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
                       :default=>"gamma_chain"))
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

function gammas()
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(γ, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for γ in 0.0:0.05:0.95]
        append!(gvfs, new_gvfs)
    end
    return Horde(gvfs)
end

function get_horde(horde_str::AbstractString)
    horde = forward()
    if horde_str == "forward"
        horde = forward()
    elseif horde_str == "rafols"
        horde = rafols()
    elseif horde_str == "gammas"
        horde = gammas()
    end
    return horde
end

get_horde(parsed::Dict, prefix::AbstractString="") = get_horde(parsed["$(prefix)horde"])

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

function oracle(env::CompassWorld, horde_str)
    
    state = env.agent_state
    ret = Array{Float64,1}()
    if horde_str == "forward"
        ret = oracle_forward(state)
    elseif horde_str == "rafols"
        oracle_rafols(state)
    elseif horde_str == "gammas"
        oracle_gammas(state)
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




end
