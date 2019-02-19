
module CycleWorld

mutable struct Environment
    pos
    size
    Environment() = new(0, 6)
end

function get_observation(env::Environment)
    if env.pos == 0
        return [1.0, 1.0, 0.0]
    else
        return [1.0, 0.0, 1.0]
    end
end

init() = Environment()

function start!(env::Environment)
    env.pos = 0
    return env
end

function step!(env_state, action)
    env_state.pos = (env_state.pos + 1) % env_state.size
    return env_state, 0, false
end

end
