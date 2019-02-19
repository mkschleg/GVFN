using Random
using StatsBase

# Abstract Policy
abstract type AbstractPolicy end

#
function get_action(policy::AbstractPolicy, state)
    throw("Implement get_action")
end

function get_action_probability(policy::AbstractPolicy, state, action)
    throw("Implement get_action_probability")
end

function get_probability_dist(policy::AbstractPolicy, state, action)
    throw("Implement get_probability_dist")
end

abstract type AbstractDistPolicy :< AbstractPolicy end

struct DiscreteDistPolicy :< AbstractDistPolicy
    dist::Array{Float64}
    range::AbstractArray
end

function get_action(policy::DiscreteDistPolicy, state; n=1, rng=Random.GLOBAL_RNG)
    # throw("Implement get_action")
    return sample(rng, policy.range, Weights(policy.dist), n)
end

function get_action_probability(policy::DiscreteDistPolicy, state, action)
    # throw("Implement get_action_probability")
    return policy.range[action]
end

function get_probability_dist(policy::DiscreteDistPolicy, state, action)
    throw("Implement get_probability_dist")
end

mutable struct ContinuousDistPolicy :< AbstractDistPolicy
    dist::Array{Float64}
end

function get_action(policy::AbstractPolicy, state)
    throw("Implement get_action")
end

function get_action_probability(policy::AbstractPolicy, state, action)
    throw("Implement get_action_probability")
end

function get_probability_dist(policy::AbstractPolicy, state, action)
    throw("Implement get_probability_dist")
end






