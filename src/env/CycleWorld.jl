
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

function JuliaRL.environment_step!(env::CycleWorld, action::Int64; rng = Random.GLOBAL_RNG, kwargs...)
    # actions 1 == Turn Left
    # actions 2 == Turn Right
    # actions 3 == Up
    # JuliaRL.step()
    env.agent_state = (env.agent_state + 1) % env.chain_length
    # JuliaRL
end

function JuliaRL.get_reward(env::CycleWorld) # -> get the reward of the environment
    return 0
end

function JuliaRL.get_state(env::CycleWorld) # -> get state of agent
    if env.partially_observable
        return partially_observable_state(env)
    else
        return fully_observable_state(env)
    end
end

function fully_observable_state(env::CycleWorld)
    return [env.agent_state]
end

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
    println(env.agent_state)
end
