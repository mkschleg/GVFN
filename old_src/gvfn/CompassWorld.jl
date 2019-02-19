
module CompassWorld

using GVFN.Environments
using GVFN
using LinearAlgebra
using Lazy

export Rafols

module Rafols

# include("../GVF.jl")

terminate_on_obs_white(obs) = obs[6] == 0.0

FORWARD_POLICY = Policy([1.0, 0.0, 0.0])
LEFT_POLICY = Policy([0.0, 1.0, 0.0])
RIGHT_POLICY = Policy([0.0, 0.0, 1.0])


function build_gvfn()
    gvfs = Array{GVF, 1}()

    for col in 1:5
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, col), Continuation(ContinuationTypes.Constant, 0.0, nothing), FORWARD_POLICY)])
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, col), Continuation(ContinuationTypes.Constant, 0.0, nothing), LEFT_POLICY)])
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, col), Continuation(ContinuationTypes.Constant, 0.0, nothing), RIGHT_POLICY)])
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, col), Continuation(ContinuationTypes.Observation, 1.0, terminate_on_obs_white), FORWARD_POLICY)])
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Composition, (col - 1)*8 + 1 + 3), Continuation(ContinuationTypes.Constant, 0.0, nothing), LEFT_POLICY)])
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Composition, (col - 1)*8 + 1 + 3), Continuation(ContinuationTypes.Constant, 0.0, nothing), RIGHT_POLICY)])
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Composition, (col - 1)*8 + 1 + 4), Continuation(ContinuationTypes.Observation, 1.0, terminate_on_obs_white), FORWARD_POLICY)])
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Composition, (col - 1)*8 + 1 + 5), Continuation(ContinuationTypes.Observation, 1.0, terminate_on_obs_white), FORWARD_POLICY)])
    end
    gvfn = GVFNetwork(gvfs, Rafols)
    return gvfn
end




const weight_dim = (8*5, (8*5 + 1 + 6)*3)
const num_gvfs = (8*5)
const γ_const = 0.9


function ORACLE(env::CompassWorld.Environment)

    truth = zeros(5)

    if env.dir == CompassWorld.NORTH
        # North
        truth[1] = 1
    elseif env.dir == CompassWorld.SOUTH
        # South
        truth[3] = 1
    elseif env.dir == CompassWorld.WEST
        # West
        if env.pos.y == 1
            truth[5] = 1
        else
            truth[4] = 1
        end
    elseif env.dir == CompassWorld.EAST
        # East
        truth[2] = 1
    else
        throw(ErrorException("Direction went wrong in ORACLE"))
    end

    return truth

end


function get_parameters!(r, γ_t, ρ_t, obs, preds_tilde)
    r[1] = obs[2]
    for i in 2:(length(preds_tilde) - 1)
        r[i] = preds_tilde[i-1]
    end
    r[end] = obs[2]
    γ_t[end] = 0.9*(1-r[1])
end

@inline sqr_error(env::CompassWorld.Environment, preds::Array{Float64}) = (preds[[4, 8+4, 16+4, 24+4, 32+4]] .- ORACLE(env)).^2
@inline sqr_error!(env::CompassWorld.Environment, preds::Array{Float64}, loc) = (loc .= (preds[[4, 8+4, 16+4, 24+4, 32+4]] .- ORACLE(env)).^2)

function make_predictions!(ϕ::Array{Float64, 1}, weights::Array{Array{Float64, 1}, 1}, preds::Array{Float64, 1}; activate=Main.sigmoid)
    # preds .= sigmoid.(dot.([ϕ], weights))
    @inbounds for gvf = 1:length(weights)
        preds[gvf] = activate(dot(ϕ, weights[gvf]))
    end
end

end

export Gamma

module Gamma

# using GVFN.Environments
# using LinearAlgebra
# using Lazy

# include("../GVF.jl")

terminate_on_obs_not_white(obs) = obs[end] == 0.0

FORWARD_POLICY = Policy([1.0, 0.0, 0.0])
LEFT_POLICY = Policy([0.0, 1.0, 0.0])
RIGHT_POLICY = Policy([0.0, 0.0, 1.0])

function forward_if_north(policy::Policy, obs, action)
    if findmax(obs[[4, 8+4, 16+4, 24+4, 32+4]])[2] == 1 && action == 1
        return 1.0
    else
        return 0.0
    end
end

function forward_if_east(policy::Policy, obs, action)
    if findmax(obs[[4, 8+4, 16+4, 24+4, 32+4]])[2] == 2 && action == 1
        return 1.0
    else
        return 0.0
    end
end

function forward_if_south(policy::Policy, obs, action)
    if findmax(obs[[4, 8+4, 16+4, 24+4, 32+4]])[2] == 3 && action == 1
        return 1.0
    else
        return 0.0
    end
end

function forward_if_west(policy::Policy, obs, action)
    mx = findmax(obs[[4, 8+4, 16+4, 24+4, 32+4]])[2]
    if (mx == 4 || mx == 5) && action == 1
        return 1.0
    else
        return 0.0
    end
end

NORTH_POLICY = Policy(PolicyTypes.Functional, [], forward_if_north)
SOUTH_POLICY = Policy(PolicyTypes.Functional, [], forward_if_south)
EAST_POLICY = Policy(PolicyTypes.Functional, [], forward_if_east)
WEST_POLICY = Policy(PolicyTypes.Functional, [], forward_if_west)


function build_gvfn()
    gvfs = Array{GVF, 1}()

    # for col in 1:5
    append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, 41), Continuation(ContinuationTypes.Observation, 0.9, terminate_on_obs_not_white), NORTH_POLICY)])
    append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, 42), Continuation(ContinuationTypes.Observation, 0.9, terminate_on_obs_not_white), EAST_POLICY)])
    append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, 43), Continuation(ContinuationTypes.Observation, 0.9, terminate_on_obs_not_white), SOUTH_POLICY)])
    append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, 44), Continuation(ContinuationTypes.Observation, 0.9, terminate_on_obs_not_white), WEST_POLICY)])
    append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, 45), Continuation(ContinuationTypes.Observation, 0.9, terminate_on_obs_not_white), WEST_POLICY)])
    # end
    gvfn = GVFNetwork(gvfs, Gamma)
    return gvfn
end

# function get_parameters!(gvfn::GVFNetwork, r, γ_t, ρ_t, obs, preds_tilde, target_policy, action)
#     tp_prob = target_policy[action]
#     for (gvf_idx, gvf) in enumerate(gvfn.gvfs)
#         r[gvf_idx] = get_cumulant(gvf, obs, preds_tilde)
#         γ_t[gvf_idx] = get_continuation(gvf, obs, preds_tilde)
#         ρ_t[gvf_idx] = get_probability(gvf, obs, action)/tp_prob
#     end
# end


const weight_dim = (5, (6 + 8*5 + 1 + 5)*3)
const γ_const = 0.9

function ORACLE(env::CompassWorld.Environment)

    truth = zeros(5)

    truth[1] = 0.9^(env.pos.y - 1) # Distance from Orange
    truth[2] = 0.9^(env.size - env.pos.x) # Distance from YELLOW
    truth[3] = 0.9^(env.size - env.pos.y)
    if env.pos.y == 1
        truth[5] = 0.9^(env.pos.x - 1)
    else
        truth[4] = 0.9^(env.pos.x - 1)
    end
    # println(truth)
    return truth

end


function get_parameters!(r, γ_t, ρ_t, obs, preds_tilde)
    r[1] = obs[2]
    for i in 2:(length(preds_tilde) - 1)
        r[i] = preds_tilde[i-1]
    end
    r[end] = obs[2]
    γ_t[end] = 0.9*(1-r[1])
end

@inline sqr_error(env::CompassWorld.Environment, preds::Array{Float64}) = (preds .- ORACLE(env)).^2
@inline sqr_error!(env::CompassWorld.Environment, preds::Array{Float64}, loc) = (loc .= (preds .- ORACLE(env)).^2)

function make_predictions!(ϕ::Array{Float64, 1}, weights::Array{Array{Float64, 1}, 1}, preds::Array{Float64, 1}; activate=Main.sigmoid)
    # preds .= sigmoid.(dot.([ϕ], weights))
    @inbounds for gvf = 1:length(weights)
        preds[gvf] = activate(dot(ϕ, weights[gvf]))
    end
end

end



export EvaluationLayer
module EvaluationLayer

# using GVFN.Environments
# using LinearAlgebra
# using Lazy

# include("../GVF.jl")

terminate_on_obs_white(obs) = obs[6] == 0.0

FORWARD_POLICY = Policy([1.0, 0.0, 0.0])
LEFT_POLICY = Policy([0.0, 1.0, 0.0])
RIGHT_POLICY = Policy([0.0, 0.0, 1.0])


function build_gvfn()
    gvfs = Array{GVF, 1}()

    for col in 1:5
        append!(gvfs, [GVF(Cumulant(CumulantTypes.Observation, col), Continuation(ContinuationTypes.Observation, 1.0, terminate_on_obs_white), FORWARD_POLICY)])
    end
    gvfn = GVFNetwork(gvfs, EvaluationLayer)
    return gvfn
end

const weight_dim = (8*5, (8*5 + 1 + 6)*3)
const γ_const = 0.9


function ORACLE(env::CompassWorld.Environment)

    truth = zeros(5)

    if env.dir == CompassWorld.NORTH
        # North
        truth[1] = 1
    elseif env.dir == CompassWorld.SOUTH
        # South
        truth[3] = 1
    elseif env.dir == CompassWorld.WEST
        # West
        if env.pos.y == 1
            truth[5] = 1
        else
            truth[4] = 1
        end
    elseif env.dir == CompassWorld.EAST
        # East
        truth[2] = 1
    else
        throw(ErrorException("Direction went wrong in ORACLE"))
    end

    return truth

end


function get_parameters!(r, γ_t, ρ_t, obs, preds_tilde)
    r[1] = obs[2]
    for i in 2:(length(preds_tilde) - 1)
        r[i] = preds_tilde[i-1]
    end
    r[end] = obs[2]
    γ_t[end] = 0.9*(1-r[1])
end

@inline sqr_error(env::CompassWorld.Environment, preds::Array{Float64}) = (preds .- ORACLE(env)).^2
@inline sqr_error!(env::CompassWorld.Environment, preds::Array{Float64}, loc) = (loc .= (preds .- ORACLE(env)).^2)

function make_predictions!(ϕ::Array{Float64, 1}, weights::Array{Array{Float64, 1}, 1}, preds::Array{Float64, 1}; activate=Main.sigmoid)
    # preds .= sigmoid.(dot.([ϕ], weights))
    @inbounds for gvf = 1:length(weights)
        preds[gvf] = activate(dot(ϕ, weights[gvf]))
    end
end
end

end
