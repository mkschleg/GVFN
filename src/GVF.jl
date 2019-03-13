using Lazy

import Base.get

abstract type AbstractParameterFunction end

function get(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde) end

function call(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde)
    get(apf::AbstractParameterFunction, state_t, action_t, state_tp1, action_tp1, preds_tilde)
end

"""
Cumulants
"""

abstract type AbstractCumulant <: AbstractParameterFunction end

function get(cumulant::AbstractCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("get(CumulantType, args...) not defined!"))
end


"""
    FeatureCumulant
    - Basic Cumulant which takes the value c_t = s_tp1[idx] for 1<=idx<=length(s_tp1)
"""
struct FeatureCumulant <: AbstractCumulant
    idx::Int
end

get(cumulant::FeatureCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1) = state_tp1[cumulant.idx]

struct PredictionCumulant <: AbstractCumulant
    idx::Int
end

get(cumulant::PredictionCumulant, state_t, action_t, state_tp1, action_tp1, preds_tp1) = preds_tp1[cumulant.idx]


"""
Discounting
"""
abstract type AbstractDiscount <: AbstractParameterFunction end

function get(γ::AbstractDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("get(DiscountType, args...) not defined!"))
end

struct ConstantDiscount{T} <: AbstractDiscount
    γ::T
end

get(cd::ConstantDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1) = cd.γ

struct StateTerminationDiscount{T<:Number, F} <: AbstractDiscount
    γ::T
    condition::F
    terminal::T
    StateTerminationDiscount(γ, condition) = new{typeof(γ), typeof(condition)}(γ, condition, convert(typeof(γ), 0.0)) 
end

get(td::StateTerminationDiscount, state_t, action_t, state_tp1, action_tp1, preds_tp1) =
    td.condition(state_tp1) ? td.terminal : td.γ


"""
Policies
"""
abstract type AbstractPolicy <: AbstractParameterFunction end

function get(π::AbstractPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("get(PolicyType, args...) not defined!"))
end


struct NullPolicy <: AbstractPolicy
end

get(π::NullPolicy, state_t, action_t, state_tp1, action_tp1, preds_tp1) = 1.0

abstract type AbstractGVF end

function get(gvf::AbstractGVF, state_t, action_t, state_tp1, action_tp1, preds_tp1) end

get(gvf::AbstractGVF, state_t, action_t, state_tp1, preds_tp1) =
    get(gvf::AbstractGVF, state_t, action_t, state_tp1, nothing, preds_tp1)

get(gvf::AbstractGVF, state_t, action_t, state_tp1) =
    get(gvf::AbstractGVF, state_t, action_t, state_tp1, nothing, nothing)

function cumulant(gvf::AbstractGVF) end
function discount(gvf::AbstractGVF) end
function policy(gvf::AbstractGVF) end

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

abstract type AbstractHorde end

struct Horde{T<:AbstractGVF} <: AbstractHorde
    gvfs::Vector{T}
end

function get(gvfh::Horde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    C = map(gvf -> get(cumulant(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Γ = map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs = map(gvf -> get(policy(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    return C, Γ, Π_probs
end

get(gvfh::Horde, state_tp1, preds_tp1) = get(gvfh::Horde, nothing, nothing, state_tp1, nothing, preds_tp1)
