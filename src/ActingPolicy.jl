

abstract type AbstractActingPolicy end

mutable struct RandomActingPolicy{T<:AbstractFloat} <: AbstractActingPolicy
    probabilities::Array{T,1}
    weight_vec::Weights{T, T, Array{T, 1}}
    RandomActingPolicy(probabilities::Array{T,1}) where {T<:AbstractFloat} =
        new{T}(probabilities, Weights(probabilities))
end

get_prob(π::RandomActingPolicy, state_t, action_t) =
    π.probabilities[action_t]

StatsBase.sample(π::RandomActingPolicy) =
    StatsBase.sample(Random.GLOBAL_RNG, π)

StatsBase.sample(rng::Random.AbstractRNG, π::RandomActingPolicy) =
    StatsBase.sample(rng, π.weight_vec)

function (π::RandomActingPolicy)(state_t, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    action = StatsBase.sample(rng, π)
    return action, get_prob(π, state_t, action)
end


mutable struct FunctionalActingPolicy{A, P}
    action_func::A
    prob_func::P
end

get_prob(π::FunctionalActingPolicy, state_t, action_t) =
    π.prob_func(π, state_t, action_t)

StatsBase.sample(π::FunctionalActingPolicy, state_t) =
    sample(Random.GLOBAL_RNG, π)

StatsBase.sample(rng::Random.AbstractRNG, π::FunctionalActingPolicy, state_t) =
    π.action_func(π, state_t, rng)

function (π::FunctionalActingPolicy)(state_t, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    action = sample(rng, π, state_t)
    return action, get_prob(π, state_t, action)
end


