using Lazy
using StatsBase

import Base.get, Base.get!

"""
    AbstractParameterFunction

An abstract type to define cumulants, discounts, and policies.
"""
abstract type AbstractParameterFunction end

function get(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde) end

function call(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde)
    get(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde)
end

"""
    AbstractCumulant

Abstract type for cumulants.
"""
abstract type AbstractCumulant <: AbstractParameterFunction end

function get(cumulant::AbstractCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("get(CumulantType, args...) not defined!"))
end


"""
    FeatureCumulant

Basic Cumulant which takes the value c_t = s_tp1[idx] for 1<=idx<=length(s_tp1)
"""
struct FeatureCumulant <: AbstractCumulant
    idx::Int
end

get(cumulant::FeatureCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1) = state_tp1[cumulant.idx]

"""
    PredictionCumulant

Cumulant which implements the compositional idea where c_t = p_tp1[idx] for 1<=idx<=length(preds_tp1).
"""
struct PredictionCumulant <: AbstractCumulant
    idx::Int
end

get(cumulant::PredictionCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1) = preds_tp1[cumulant.idx]

"""
    ScaledCumulant

A cumulant which is scaled by a number. This scales the result of get(cumulant, ...)
"""
struct ScaledCumulant{F<:Number, T<:AbstractCumulant} <: AbstractCumulant
    scale::F
    cumulant::T
end

get(cumulant::ScaledCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1) =
    cumulant.scale*get(cumulant.cumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)

"""
    NormalizedCumulant

This scales a cumulant by c * scale / rmax where rmax is calculated online or passed in the constructor.
"""
mutable struct NormalizedCumulant{F<:Number, T<:AbstractCumulant} <: AbstractCumulant
    scale::F
    cumulant::T
    rmax::F
end

NormalizedCumulant(scale, cumulant) = NormalizedCumulant(scale, cumulant, 1.0f0)

function get(cumulant::NormalizedCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    c = get(cumulant.cumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    cumulant.rmax = max(cumulant.rmax,c)
    return c * cumulant.scale / cumulant.rmax
end

"""
    UnityCumulant

This scales the cumulant to be in the range 0.0-1.0.
"""
mutable struct UnityCumulant{F<:Number, T<:AbstractCumulant} <: AbstractCumulant
    scale::F
    cumulant::T
    rmax::F
    rmin::F
end

UnityCumulant(scale, cumulant) = UnityCumulant(scale, cumulant, 1.0f0, 0.0f0)

function get(cumulant::UnityCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    c = get(cumulant.cumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    cumulant.rmax, cumulant.rmin = max(cumulant.rmax,c), min(cumulant.rmin, c)
    return (c - cumulant.rmin) * (cumulant.rmax-cumulant.rmin) * cumulant.scale 
end

"""
    AbstractDiscount

Abstract type for discounts.
"""
abstract type AbstractDiscount <: AbstractParameterFunction end

function get(γ::AbstractDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("get(DiscountType, args...) not defined!"))
end

"""
    ConstantDiscount

Will always return γ
"""
struct ConstantDiscount{T} <: AbstractDiscount
    γ::T
end

get(cd::ConstantDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1) = cd.γ

"""
    StateTerminationDiscount

Returns 0.0 if condition(state_tp1) is true, otherwise it returns γ.
"""
struct StateTerminationDiscount{T<:Number, F} <: AbstractDiscount
    γ::T
    condition::F
    terminal::T
    StateTerminationDiscount(γ, condition) = new{typeof(γ), typeof(condition)}(γ, condition, convert(typeof(γ), 0.0f0)) 
end

get(td::StateTerminationDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1) =
    td.condition(state_tp1) ? td.terminal : td.γ


"""
    AbstractPolicy

"""
abstract type AbstractPolicy <: AbstractParameterFunction end

function get(π::AbstractPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("get(PolicyType, args...) not defined!"))
end

"""
    NullPolicy

Stand in policy when it isn't needed. Always returns 1.0f0.
"""
struct NullPolicy <: AbstractPolicy
end
get(π::NullPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) = 1.0f0

"""
    PersistentPolicy

Returns 1.0f0 if action == action_t and 0.0f0 otherwise.
"""
struct PersistentPolicy <: AbstractPolicy
    action::Int64
end

get(π::PersistentPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) = π.action == action_t ? 1.0f0 : 0.0f0

"""
    RandomPolicy

Returns the probability of taking action_t based on probabilities. This type of policy can also be sampled from.
"""
struct RandomPolicy{T<:AbstractFloat} <: AbstractPolicy
    probabilities::Array{T,1}
    weight_vec::Weights{T, T, Array{T, 1}}
    RandomPolicy(probabilities::Array{T,1}) where {T<:AbstractFloat} = new{T}(probabilities, Weights(probabilities))
end

get(π::RandomPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) = π.probabilities[action_t]

StatsBase.sample(π::RandomPolicy) = StatsBase.sample(Random.GLOBAL_RNG, π)
StatsBase.sample(rng::Random.AbstractRNG, π::RandomPolicy) = StatsBase.sample(rng, π.weight_vec)
StatsBase.sample(rng::Random.AbstractRNG, π::RandomPolicy, state) = StatsBase.sample(rng, π.weight_vec)

"""
    FunctionalPolicy

Returns func(state_t, action_t, state_tp1, action_tp1, preds_tp1)
"""
struct FunctionalPolicy{F} <: AbstractPolicy
    func::F
end

Base.get(π::FunctionalPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) =
    π.func(state_t, action_t, state_tp1, action_tp1, preds_tp1)

"""
    PredictionConditionalPolicy

Returns condition(preds_tp1) * get(policy, ...)
"""
struct PredictionConditionalPolicy{P<:AbstractPolicy, F} <: AbstractPolicy
    policy::P
    condition::F
end

Base.get(π::PredictionConditionalPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) =
    π.condition(preds_tp1) * get(π.policy, state_t, action_t, state_tp1, action_tp1, preds_tp1)

"""
    AbstractGVF
"""
abstract type AbstractGVF end

function get(gvf::AbstractGVF, state_t, action_t, state_tp1, action_tp1, preds_tp1) end

get(gvf::AbstractGVF, state_t, action_t, state_tp1, preds_tp1) =
    get(gvf::AbstractGVF, state_t, action_t, state_tp1, nothing, preds_tp1)

get(gvf::AbstractGVF, state_t, action_t, state_tp1) =
    get(gvf::AbstractGVF, state_t, action_t, state_tp1, nothing, nothing)

function cumulant(gvf::AbstractGVF) end
function discount(gvf::AbstractGVF) end
function policy(gvf::AbstractGVF) end

"""
    GVF

Basic provided GVF. Expects a Cumulant, Discount, and Policy all of which are derived from the above abstract types.
"""
struct GVF{C<:AbstractCumulant, D<:AbstractDiscount, P<:AbstractPolicy} <: AbstractGVF
    cumulant::C
    discount::D
    policy::P
end

cumulant(gvf::GVF) = gvf.cumulant
discount(gvf::GVF) = gvf.discount
policy(gvf::GVF) = gvf.policy

function get(gvf::GVF, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    c = get(gvf.cumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    γ = get(gvf.discount, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    π_prob = get(gvf.policy, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    return c, γ, π_prob
end

"""
    AbstractHorde
"""
abstract type AbstractHorde end

struct Horde{T<:AbstractGVF} <: AbstractHorde
    gvfs::Vector{T}
end

merge(horde1::Horde, horde2::Horde) = Horde([horde1.gvfs; horde2.gvfs])


# combine(gvfh_1::Horde, gvfh_2::Horde) = Horde([gvfh_1.gvfs; ])

function get(gvfh::Horde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    C = Float32.(map(gvf -> get(cumulant(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs))
    Γ = Float32.(map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs))
    Π_probs = Float32.(map(gvf -> get(policy(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs))
    return C, Γ, Π_probs
end

function Base.get!(C::Array{T, 1}, Γ::Array{T, 1}, Π_probs::Array{T, 1}, gvfh::Horde, state_t, action_t, state_tp1, action_tp1, preds_tp1) where {T<:AbstractFloat}
    C .= map(gvf -> get(cumulant(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Γ .= map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs .= map(gvf -> get(policy(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    return C, Γ, Π_probs
end

get(gvfh::Horde, state_tp1, preds_tp1) = get(gvfh::Horde, nothing, nothing, state_tp1, nothing, preds_tp1)
get(gvfh::Horde, action_t, state_tp1, preds_tp1) = get(gvfh::Horde, nothing, action_t, state_tp1, nothing, preds_tp1)
get(gvfh::Horde, state_t, action_t, state_tp1, preds_tp1) = get(gvfh::Horde, state_t, action_t, state_tp1, nothing, preds_tp1)

get!(C, Γ, Π_probs,gvfh::Horde, action_t, state_tp1, preds_tp1) = get!(C, Γ, Π_probs, gvfh::Horde, nothing, action_t, state_tp1, nothing, preds_tp1)

@forward Horde.gvfs Base.length, Base.getindex




