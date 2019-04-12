
using Random
# using JuliaRL

# import JuliaRL.reset!, JuliaRL.environment_step!, JuliaRL.get_reward

"""
 CycleWorld

   1 -> 0 -> 0 -> ... -> 0 -|
   ^------------------------|

chain_length: size of cycle
actions: Progress

"""

mutable struct CycleWorld <: AbstractEnvironment
    chain_length::Int64
    agent_state::Int64
    actions::AbstractSet
    partially_observable::Bool
    CycleWorld(chain_length::Int64; rng=Random.GLOBAL_RNG, partially_observable=true) =
        new(chain_length,
            0,
            Set(1:1),
            partially_observable)
end

function JuliaRL.reset!(env::CycleWorld; rng = Random.GLOBAL_RNG, kwargs...)
    env.agent_state = 0
end

JuliaRL.get_actions(env::CycleWorld) = env.actions

JuliaRL.environment_step!(env::CycleWorld, action::Int64; rng = Random.GLOBAL_RNG, kwargs...) = 
    env.agent_state = (env.agent_state + 1) % env.chain_length


JuliaRL.get_reward(env::CycleWorld) = 0 # -> get the reward of the environment

function JuliaRL.get_state(env::CycleWorld) # -> get state of agent
    if env.partially_observable
        return partially_observable_state(env)
    else
        return fully_observable_state(env)
    end
end

fully_observable_state(env::CycleWorld) = [env.agent_state]

function partially_observable_state(env::CycleWorld)
    state = zeros(1)
    if env.agent_state == 0
        state[1] = 1
    end
    return state
end

function JuliaRL.is_terminal(env::CycleWorld) # -> determines if the agent_state is terminal
    return false
end

function Base.show(io::IO, env::CycleWorld)
    model = fill("0", env.chain_length)
    model[1] = "1"
    println(model)
    model = fill("-", env.chain_length)
    model[env.agent_state + 1] = "a"
    println(model)
    # println(env.agent_state)
end



