

abstract type AbstractActingPolicy end


mutable struct RandomActingPolicy{T<:AbstractFloat}
    probabilities::Array{T,1}
    weight_vec::Weights{T, T, Array{T, 1}}
    RandomActingPolicy{T}(probabilities::Array{T,1}) where {T<:AbstractFloat} =
        new{T}(probabilities, Weights(probabilities))
end

get(π::RandomActingPolicy, state_t, action_t) =
    π.probabilities[action_t]

StatsBase.sample(π::RandomActingPolicy) =
    sample(Random.GLOBAL_RNG, π)

StatsBase.sample(rng::Random.AbstractRNG, π::RandomActingPolicy) =
    sample(rng, π.weight_vec, 1:length(π.weight_vec))


mutable struct FunctionalActingPolicy{A, P}
    action_func::A
    prob_func::P
end

get(π::FunctionalActingPolicy, state_t, action_t) =
    π.prob_func(π, state_t, action_t)

StatsBase.sample(π::FunctionalActingPolicy, state_t) =
    sample(Random.GLOBAL_RNG, π)

StatsBase.sample(rng::Random.AbstractRNG, π::FunctionalActingPolicy, state_t) =
    π.action_func(π, state_t, rng)
