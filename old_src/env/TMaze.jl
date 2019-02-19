module CycleWorld

using Random

@enum GOAL_STATES begin
    UP = 1
    DOWN
end

const num_actions = 4
const action_space = 1:4

const negative_reward = -0.1
const positive_reward = 4.0

mutable struct Environment
    pos::Int64
    size::Int64
    goal_state::GOAL_STATES
    Environment() = new(0, 10, UP)
end

function get_observation(env::Environment)
    if env.pos == 1
        if goal_state == UP
            return [1.0, 1.0, 0.0]
        else
            return [0.0, 1.0, 1.0]
        end
    elseif env.pos == size
        return [0.0, 1.0, 0.0]
    else
        return [1.0, 0.0, 1.0]
    end
end

init() = Environment()

function start!(env::Environment; rng=Random.GLOBAL_RNG)
    env.pos = 1
    env.goal_state = rand(rng, [UP, DOWN])
    return env
end

function step!(env_state, action)
    rew = 0.0
    term = false
    prev_pos = env_state.pos

    if action == 1
        # up
        if env_state.pos == size
            term = true
            if env_state.goal_state == UP
                rew = positive_reward
            else
                rew = negative_reward
            end
        end
    elseif action == 2
        # right
        env_state.pos = env_state.pos + 1
    elseif action == 3
        # down
        if env_state.pos == size
            term = true
            if env_state.goal_state == UP
                rew = negative_reward
            else
                rew = positive_reward
            end
        end
    elseif action == 4
        # left
        env_state.pos = env_state.pos - 1
    end

    env_state.pos = clamp(env_state.pos, 1, env_state.size)
    if env_state.pos == prev_pos && prev_pos == 1
        rew = -0.1
    end


    return env_state, rew, term
end

end
