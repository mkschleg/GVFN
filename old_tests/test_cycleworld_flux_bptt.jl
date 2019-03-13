
using Flux
using Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random

module CycleWorld
using Random

"""
 CycleWorld

   1 -> 0 -> 0 -> ... -> 0 -|
   ^------------------------|

chain_length: size of cycle
actions: Progress

"""

mutable struct CycleWorldEnv
    chain_length::Int64
    agent_state::Int64
    actions::AbstractSet
    partially_observable::Bool
    CycleWorldEnv(chain_length::Int64; rng=Random.GLOBAL_RNG, partially_observable=true) =
        new(chain_length,
            0,
            Set(1:1),
            partially_observable)
end

function reset!(env::CycleWorldEnv; rng = Random.GLOBAL_RNG, kwargs...)
    env.agent_state = 0
end

get_actions(env::CycleWorldEnv) = env.actions

function environment_step!(env::CycleWorldEnv, action::Int64; rng = Random.GLOBAL_RNG, kwargs...)
    # actions 1 == Turn Left
    # actions 2 == Turn Right
    # actions 3 == Up
    # JuliaRL.step()
    env.agent_state = (env.agent_state + 1) % env.chain_length
    # JuliaRL
end

function get_reward(env::CycleWorldEnv) # -> get the reward of the environment
    return 0
end

function get_state(env::CycleWorldEnv) # -> get state of agent
    if env.partially_observable
        return partially_observable_state(env)
    else
        return fully_observable_state(env)
    end
end

function fully_observable_state(env::CycleWorldEnv)
    return [env.agent_state]
end

function partially_observable_state(env::CycleWorldEnv)
    state = zeros(1)
    if env.agent_state == 0
        state[1] = 1
    end
    return state
end

function is_terminal(env::CycleWorldEnv) # -> determines if the agent_state is terminal
    return false
end

function Base.show(io::IO, env::CycleWorldEnv)
    println(env.agent_state)
end

function start!(env::CycleWorldEnv; rng=Random.GLOBAL_RNG, kwargs...)
    reset!(env; rng=rng, kwargs...)
    return env, get_state(env)
end

function step!(env::CycleWorldEnv, action; rng = Random.GLOBAL_RNG, kwargs...) # -> env, state, reward, terminal
    environment_step!(env, action; rng=rng, kwargs...)
    return env, get_state(env), get_reward(env), is_terminal(env)
end


end #end CycleWorld




build_features(s) = [1.0, s[1], 1-s[1]]

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


abstract type AbstractGVFLayer end

function get_question_parameters(gvfn::AbstractGVFLayer, preds_tilde, stp1) end

mutable struct GVFLayer{F, A, V} <: AbstractGVFLayer
    σ::F
    Wx::A
    Wh::A
    cumulants::AbstractArray
    discounts::AbstractArray
    h::V
end

GVFLayer(num_gvfs, num_ext_features, cumulants, discounts; init=Flux.glorot_uniform, σ_int=σ) =
    GVFLayer(
        σ_int,
        param(0.1.*init(num_gvfs, num_ext_features)),
        param(0.1.*init(num_gvfs, num_gvfs)),
        cumulants,
        discounts,
        param(Flux.zeros(num_gvfs)))

function get_question_parameters(gvfn::GVFLayer{F,A,V}, preds_tilde, state_tp1) where {F, A, V}
    cumulants = [gvfn.cumulants[i](state_tp1, preds_tilde) for i in 1:length(gvfn.cumulants)]
    discounts = [gvfn.discounts[i](state_tp1) for i in 1:length(gvfn.cumulants)]
    return cumulants, discounts, ones(size(gvfn.discounts))
end

function (m::GVFLayer)(h, x)
    new_h = m.σ.(m.Wx*x + m.Wh*h)
    return new_h, new_h
end



Flux.hidden(m::GVFLayer) = m.h
Flux.@treelike GVFLayer
GVFN(args...; kwargs...) = Flux.Recur(GVFLayer(args...; kwargs...))

function reset!(m, h_init)
    Flux.reset!(m)
    m.state.data .= h_init
end

function jacobian(δ, pms)
    k  = length(δ)
    J = IdDict()
    for id in pms
        v = get!(J, id, zeros(k, size(id)...))
        for i = 1:k
            Flux.back!(δ[i], once = false) # Populate gradient accumulator
            v[i, :,:] .= id.grad
            id.grad .= 0 # Reset gradient accumulator
        end
    end
    J
end

function jacobian!(J::IdDict, δ::TrackedArray, pms::Params)
    k  = length(δ)
    for i = 1:k
        Flux.back!(δ[i], once = false) # Populate gradient accumulator
        for id in pms
            v = get!(J, id, zeros(typeof(id[1].data), k, size(id)...))::Array{typeof(id[1].data), 3}
            v[i, :, :] .= id.grad
            id.grad .= 0 # Reset gradient accumulator
        end
    end
end

abstract type Optimizer end

function train!(gvfn::Flux.Recur{T}, opt::Optimizer, h_init, state_seq, env_state_tp1) where {T <: AbstractGVFLayer} end
function train!(gvfn::AbstractGVFLayer, opt::Optimizer, h_init, state_seq, env_state_tp1)
    throw("$(typeof(opt)) not implemented for $(typeof(gvfn)). Try Flux.Recur{$(typeof(gvfn))} ")
end

mutable struct RTD_jacobian <: Optimizer
    α::Float64
    J::IdDict
    RTD_jacobian(α) = new(α, IdDict{Any, Array{Float64, 3}}())
end

function train!(gvfn::Flux.Recur{T}, opt::RTD_jacobian, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    α = opt.α
    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, preds_tilde, env_state_tp1)


    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    jacobian!(opt.J, δ, Params([gvfn.cell.Wx, gvfn.cell.Wh]))

    for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
        Flux.Tracker.update!(weights, -α.*sum(δ.*opt.J[weights]; dims=1)[1,:,:])
    end
end

mutable struct RTD <: Optimizer
    α::Float64
    RTD(α) = new(α)
end

function train!(gvfn::Flux.Recur{T}, opt::RTD, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    α = opt.α

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, preds_tilde, env_state_tp1)
    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    prms = params(gvfn)
    # println(prms)
    grads = Tracker.gradient(() ->mean(δ.^2), prms)
    for weights in prms
        Flux.Tracker.update!(weights, -α.*grads[weights])
    end
end

mutable struct RTDC <: Optimizer
    α::Float64
    β::Float64
    h::IdDict
    RTD(α) = new(α)
end

function train!(gvfn::Flux.Recur{T}, opt::RTDC, h_init, states, env_state_tp1) where {T <: AbstractGVFLayer}

    α = opt.α

    reset!(gvfn, h_init)
    preds = gvfn.(states)
    preds_t = preds[end-1]
    preds_tilde = Flux.data(preds[end])

    cumulants, discounts, ρ = get_question_parameters(gvfn.cell, preds_tilde, env_state_tp1)
    targets = cumulants .+ discounts.*preds_tilde
    δ = targets .- preds_t

    prms = params(gvfn)
    # println(prms)
    grads = Tracker.gradient(() ->mean(δ.^2), prms)
    for weights in prms
        Flux.Tracker.update!(weights, -α.*grads[weights])
    end
end


function test_GVFN_bptt()

    env = CycleWorld.CycleWorldEnv(6)
    num_gvfs = 6
    num_steps = 50000
    τ=7

    rng = Random.MersenneTwister(10)

    opt = RTD(0.6)
    discount = (args...)->0.0
    cumulant(i) = (s_tp1, p_tp1)-> i==1 ? s_tp1[1] : p_tp1[i-1]
    # cumulants = [[cumulant(i) for i in 1:6]; [cumulant(1)]]
    # discounts = [[discount for i in 1:6]; [(env_state)-> env_state[1] == 1 ? 0.0 : 0.9]]

    cumulants = [cumulant(i) for i in 1:6]
    discounts = [discount for i in 1:6]

    pred_strg = zeros(num_steps, num_gvfs)

    gvfn = GVFN(num_gvfs, 3, cumulants, discounts; init=(dims...)->glorot_uniform(rng, dims...))
    _, s_t = CycleWorld.start!(env)
    h_t = zeros(num_gvfs)
    h_tm1 = zeros(num_gvfs)

    state_list = [zeros(3) for t in 1:τ]
    popfirst!(state_list)
    push!(state_list, build_features(s_t))
    hidden_state_init = zeros(num_gvfs)

    for step in 1:num_steps
        _, s_tp1, _, _ = CycleWorld.step!(env, 1)

        if length(state_list) == (τ+1)
            # println(state_list)
            popfirst!(state_list)
        end
        push!(state_list, build_features(s_tp1))

        print(step, "\r")

        train!(gvfn, opt, hidden_state_init, state_list, s_tp1)

        reset!(gvfn, hidden_state_init)
        preds = gvfn.(state_list)
        pred_strg[step,:] .= preds[end].data
        # println(env.agent_state)
        # println(preds[end])
        s_t .= s_tp1
        hidden_state_init .= Flux.data(preds[1])
        # println(preds)
        # println(hidden_state_init)

    end
    return pred_strg

end



# mutable struct TDLambda <: Optimizer
#     α::Float32
#     λ::Float32
#     traces::Flux.IdDict
# end

# function train!(gvfn::Flux.Recur{GVFLayer{F,A,V}}, opt::TDLambda, h_init, state_seq, env_state_tp1) where {F, A, V}


#     reset!(gvfn, h_init)
#     # println(states)
#     preds = gvfn.(states)
#     # println(preds)
#     preds_t = preds[end-1]
#     preds_tilde = Flux.data(preds[end])

#     cumulants = [gvfn.cell.cumulants[i](env_state_tp1, preds_tilde) for i in 1:length(gvfn.cell.cumulants)]
#     discounts = [gvfn.cell.discounts[i](env_state_tp1) for i in 1:length(gvfn.cell.cumulants)]

#     targets = cumulants .+ discounts.*preds_tilde
#     δ = targets .- preds_t

#     grads = Tracker.gradient(() ->mean(δ.^2), Params([gvfn.cell.Wx, gvfn.cell.Wh]))

#     for weights in Params([gvfn.cell.Wx, gvfn.cell.Wh])
#         Flux.Tracker.update!(weights, -α.*(grads[weights]))
#     end


# end
