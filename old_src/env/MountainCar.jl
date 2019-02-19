

module MountainCar

using Random

const vel_limit = (-0.07, 0.07)
const pos_limit = (-1.2, 0.5)
const pos_initial_range = (-0.6, 0.4)


mutable struct State
    pos::Float64
    vel::Float64
    State() = new(0.0, 0.0)
    State(rng::AbstractRNG) = new(rand(rng)*(pos_initial_range[2] - pos_initial_range[1]) + pos_initial_range[1], 0)
end

function get_observation(state::State)
    return [state.pos]
end

function start(;rng = Random.GLOBAL_RNG)
    return State(rng)
end

function step!(state::State, action; kwargs...) # -> agent_state, reward, terminal
    # println(clamp(agent_state.vel + action*0.001 - 0.0025*cos(3*agent_state.pos), vel_limit...))
    state.vel = clamp(state.vel + (action - 1)*0.001 - 0.0025*cos(3*state.pos), vel_limit...)
    state.pos = clamp(state.pos + state.vel, pos_limit...)
    return state, get_reward(state), is_terminal(state)
end

function step(state::State, action; kwargs...)
    new_state = copy(state)
    return step!(new_agent_state, action; kwargs...)
end

function get_reward(state) # -> determines if the agent_state is terminal
    if state.pos >= pos_limit[2]
        return 0
    end
    return -1
end

function is_terminal(state) # -> determines if the agent_state is terminal
    return state.pos >= pos_limit[2]
end

function normalized_features(state)
    return [(state.pos - pos_limit[1])/(pos_limit[2] - pos_limit[1]), (state.vel - vel_limit[1])/(vel_limit[2] - vel_limit[1])]
end

end
