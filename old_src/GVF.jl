
using Lazy

module CumulantTypes
@enum enum begin
    Observation = 1
    Composition = 2
    Reward = 3
end
end

mutable struct Cumulant
    typ::CumulantTypes.enum
    dependent::Int64
end

function get_cumulant(cumulant::Cumulant, obs, preds_tilde)
    if cumulant.typ == CumulantTypes.Observation
        return obs[cumulant.dependent]
    elseif cumulant.typ == CumulantTypes.Composition
        # println("Composition")
        return preds_tilde[cumulant.dependent]
    else
        return 0
    end
end

# const ContinuationType = ["Observation", "Constant"]
module ContinuationTypes
@enum enum begin
    Observation = 1
    Constant = 2
    Myopic = 3
    Functional = 4
end
end

mutable struct Continuation
    typ::ContinuationTypes.enum
    γ_const::Float64
    terminate
    func
    Continuation(typ, γ_const, terminate) = new(typ, γ_const, terminate, nothing)
    Continuation(typ, γ_const, terminate, func) = new(typ, γ_const, terminate, func)
end

function get_continuation(continuation::Continuation, obs, preds_tilde)
    if continuation.typ == ContinuationTypes.Observation
        if continuation.terminate(obs)
            return 0.0
        else
            return continuation.γ_const
        end
    elseif continuation.typ == ContinuationTypes.Constant
        return continuation.γ_const
    elseif continuation.typ == ContinuationTypes.Functional
        return continuation.func(continuation, obs, preds_tilde)
    else
        return 0.0
    end
end

module PolicyTypes
@enum enum begin
    Probabilities
    Functional
end
end

mutable struct Policy
    typ::PolicyTypes.enum
    probabilities::Array{Float64, 1}
    func
end

Policy(probabilities::Array{Float64,1}) = Policy(PolicyTypes.Probabilities, probabilities, nothing)

function get_probability(policy::Policy, state, action)
    if policy.typ == PolicyTypes.Probabilities
        return policy.probabilities[action]
    elseif policy.typ == PolicyTypes.Functional
        return policy.func(policy, state, action)
    else
        throw(ErrorException("Policy Type doesn't exist!"))
    end
end


mutable struct GVF
    cumulant::Cumulant
    continuation::Continuation
    policy::Policy
end

@forward GVF.cumulant get_cumulant
@forward GVF.continuation get_continuation
@forward GVF.policy get_probability

