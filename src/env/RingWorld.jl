
using Random
using MinimalRLCore

# import JuliaRL.reset!, JuliaRL.environment_step!, JuliaRL.get_reward



module RingWorldConst

const FORWARD = 1
const BACKWARD = 2
const ACTIONS = [FORWARD, BACKWARD]

end


"""
 RingWorld
    States: 1     2     3     ...     n
    Obs:    1 <-> 0 <-> 0 <-> ... <-> 0 <-|
            ^-----------------------------|

ring_size: size (diameter) of ring
actions: Forward of Backward (See RingWorldConst)

"""
mutable struct RingWorld <: AbstractEnvironment
    ring_size::Int
    agent_state::Int
    partially_observable::Bool
end

RingWorld(ring_size::Int; partially_observable=true) =
    RingWorld(ring_size,
              1,
              partially_observable)

function env_settings!(as::Reproduce.ArgParseSettings,
                       env_type::Type{RingWorld})
    Reproduce.@add_arg_table as begin
        "--size"
        help="The length of the ring world chain"
        arg_type=Int64
        default=6
    end
end

Base.size(env::RingWorld) = env.ring_size

function MinimalRLCore.reset!(env::RingWorld, rng = Random.GLOBAL_RNG)
    env.agent_state = rand(rng, 1:size(env))
end

MinimalRLCore.get_actions(env::RingWorld) = RingWorldConst.ACTIONS

@inline take_forward_step(env::RingWorld, action::Int) =
    env.agent_state == size(env) ? 1 : env.agent_state + 1

@inline take_backward_step(env::RingWorld, action::Int) =
    env.agent_state == 1 ? size(env) : env.agent_state - 1

function MinimalRLCore.environment_step!(env::RingWorld, action::Int, rng = Random.GLOBAL_RNG)

    rwc = RingWorldConst
    
    if rwc.FORWARD == action
        env.agent_state = take_forward_step(env, action)
    elseif rwc.BACKWARD == action
        env.agent_state = take_backward_step(env, action)
    else
        throw("Action $(action) is not available!")
    end

end

function MinimalRLCore.get_reward(env::RingWorld) # -> get the reward of the environment
    return 0.0f0
end

function MinimalRLCore.get_state(env::RingWorld) # -> get state of agent
    if env.partially_observable
        return partially_observable_state(env)
    else
        return fully_observable_state(env)
    end
end

function fully_observable_state(env::RingWorld)
    return [env.agent_state]
end

function partially_observable_state(env::RingWorld)
    state = zeros(1)
    if env.agent_state == 1
        state[1] = 1
    end
    return state
end

function MinimalRLCore.is_terminal(env::RingWorld) # -> determines if the agent_state is terminal
    return false
end

function Base.string(env::RingWorld)
    model = fill("0", size(env))
    model[1] = "1"
    env_str = "   Env: \t" * join(model, ' ')
    fill!(model, "-")
    model[env.agent_state] = "a"
    agent_str = "   Agent: \t" * join(model, ' ')
    return "RingWorld:\n" * env_str * "\n" * agent_str 
end

